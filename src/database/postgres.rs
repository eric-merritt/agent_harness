// PostgreSQL connection and query helpers for conversations & messages.

use std::sync::Arc;
use sqlx::{PgPool, Error as SqlxError, Row};
use uuid::Uuid;

/// A conversation stored in the database.
#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A single message within a conversation.
#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub role: String,
    pub content: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Database manager for conversations and messages.
#[derive(Clone)]
pub struct Database {
    pool: Arc<PgPool>,
}

impl Database {
    /// Create a new Database instance connected to the given PostgreSQL URL.
    pub async fn new(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }

    /// Run the schema migration to create tables if they don't exist.
    pub async fn init(&self) -> Result<(), SqlxError> {
        log::info!("Database init: creating schema tables");

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            "#,
        )
        .execute(&*self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            "#,
        )
        .execute(&*self.pool)
        .await?;

        // ── Web tool: page content store ────────────────────────────────────
        // Fetched page content (stripped, Pug-ified) stored by short ref.
        // Agent never sees raw HTML — only gets the ref and hydrates on demand.
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS page_content (
                ref_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                pug_content TEXT NOT NULL,
                raw_html BYTEA,
                content_type TEXT,
                has_cf_challenge BOOLEAN DEFAULT FALSE,
                media_links JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
            "#,
        )
        .execute(&*self.pool)
        .await?;

        // ── Web tool: summary ref store ─────────────────────────────────────
        // Generic JSON payloads (search results, structured summaries, etc.)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS summary_refs (
                ref_id TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
            "#,
        )
        .execute(&*self.pool)
        .await?;

        // ── Web tool: cookie jar ────────────────────────────────────────────
        // Persistent cookies per domain for authenticated sessions
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cookies (
                id SERIAL PRIMARY KEY,
                domain TEXT NOT NULL,
                name TEXT NOT NULL,
                value TEXT NOT NULL,
                path TEXT DEFAULT '/',
                expires_at TIMESTAMP WITH TIME ZONE,
                is_secure BOOLEAN DEFAULT TRUE,
                is_httponly BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            "#,
        )
        .execute(&*self.pool)
        .await?;

        // ── Web tool: download jobs ─────────────────────────────────────────
        // Track background file downloads
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS download_jobs (
                job_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                dest_path TEXT NOT NULL,
                total_bytes BIGINT,
                downloaded_bytes BIGINT DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                completed_at TIMESTAMP WITH TIME ZONE
            )
            "#,
        )
        .execute(&*self.pool)
        .await?;

        log::info!("Database init: schema tables created successfully");

        Ok(())
    }

    /// Store fetched page content and return a short ref ID.
    pub async fn store_page_content(
        &self,
        url: &str,
        pug_content: &str,
        raw_html: Option<Vec<u8>>,
        content_type: Option<&str>,
        has_cf_challenge: bool,
        media_links: serde_json::Value,
    ) -> Result<String, SqlxError> {
        use sha2::{Sha256, Digest};

        // Generate a deterministic-ish short ref from URL + timestamp
        let now = chrono::Utc::now();
        let entropy = format!("{}-{}-{}", url, now.timestamp_nanos_opt().unwrap_or(0), uuid::Uuid::new_v4());
        let hash = Sha256::digest(entropy.as_bytes());
        let ref_id = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            &hash[..8],
        )[..8].to_string();

        let expires_at = now + chrono::Duration::hours(24);

        sqlx::query(
            r#"
            INSERT INTO page_content (ref_id, url, pug_content, raw_html, content_type, has_cf_challenge, media_links, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (ref_id) DO UPDATE SET
                pug_content = EXCLUDED.pug_content,
                raw_html = EXCLUDED.raw_html,
                content_type = EXCLUDED.content_type,
                has_cf_challenge = EXCLUDED.has_cf_challenge,
                media_links = EXCLUDED.media_links,
                expires_at = EXCLUDED.expires_at
            "#,
        )
        .bind(&ref_id)
        .bind(url)
        .bind(pug_content)
        .bind(raw_html)
        .bind(content_type)
        .bind(has_cf_challenge)
        .bind(media_links)
        .bind(expires_at)
        .execute(&*self.pool)
        .await?;

        // Sweep expired rows
        sqlx::query("DELETE FROM page_content WHERE expires_at < NOW()")
            .execute(&*self.pool)
            .await?;

        log::info!("Database: stored page content for url='{}', ref_id='{}'", url, ref_id);

        Ok(ref_id)
    }

    /// Hydrate page content by ref.
    pub async fn get_page_content(
        &self,
        ref_id: &str,
    ) -> Result<Option<(String, String, serde_json::Value, bool)>, SqlxError> {
        let row = sqlx::query(
            r#"
            SELECT url, pug_content, media_links, has_cf_challenge
            FROM page_content
            WHERE ref_id = $1
            "#,
        )
        .bind(ref_id)
        .fetch_optional(&*self.pool)
        .await?;

        Ok(row.map(|r| (
            r.get::<String, _>(0),   // url
            r.get::<String, _>(1),   // pug_content
            r.get::<serde_json::Value, _>(2), // media_links
            r.get::<bool, _>(3),     // has_cf_challenge
        )))
    }

    /// Store a generic JSON payload and return a ref.
    pub async fn store_summary_ref(
        &self,
        payload: &serde_json::Value,
    ) -> Result<String, SqlxError> {
        use sha2::{Sha256, Digest};

        let now = chrono::Utc::now();
        let entropy = format!("{}-{}", payload, now.timestamp_nanos_opt().unwrap_or(0));
        let hash = Sha256::digest(entropy.as_bytes());
        let ref_id = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            &hash[..8],
        )[..8].to_string();

        let expires_at = now + chrono::Duration::days(7);

        sqlx::query(
            r#"
            INSERT INTO summary_refs (ref_id, data, expires_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (ref_id) DO UPDATE SET
                data = EXCLUDED.data,
                expires_at = EXCLUDED.expires_at
            "#,
        )
        .bind(&ref_id)
        .bind(payload)
        .bind(expires_at)
        .execute(&*self.pool)
        .await?;

        sqlx::query("DELETE FROM summary_refs WHERE expires_at < NOW()")
            .execute(&*self.pool)
            .await?;

        log::info!("Database: stored summary ref_id='{}'", ref_id);

        Ok(ref_id)
    }

    /// Load a summary ref by ID.
    pub async fn load_summary_ref(
        &self,
        ref_id: &str,
    ) -> Result<Option<serde_json::Value>, SqlxError> {
        let row = sqlx::query(
            r#"SELECT data FROM summary_refs WHERE ref_id = $1"#,
        )
        .bind(ref_id)
        .fetch_optional(&*self.pool)
        .await?;

        Ok(row.and_then(|r| r.try_get::<serde_json::Value, _>(0).ok()))
    }

    /// Store cookies for a domain.
    pub async fn store_cookies(
        &self,
        domain: &str,
        cookies: &[(String, String, Option<chrono::DateTime<chrono::Utc>>)],
    ) -> Result<(), SqlxError> {
        // Delete expired first
        sqlx::query("DELETE FROM cookies WHERE expires_at IS NOT NULL AND expires_at < NOW()")
            .execute(&*self.pool)
            .await?;

        for (name, value, expires_at) in cookies {
            sqlx::query(
                r#"
                INSERT INTO cookies (domain, name, value, expires_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (domain, name) DO UPDATE SET
                    value = EXCLUDED.value,
                    expires_at = EXCLUDED.expires_at
                "#,
            )
            .bind(domain)
            .bind(name)
            .bind(value)
            .bind(expires_at)
            .execute(&*self.pool)
            .await?;
        }

        Ok(())
    }

    /// Get all cookies for a domain.
    pub async fn get_cookies(
        &self,
        domain: &str,
    ) -> Result<Vec<(String, String)>, SqlxError> {
        let rows = sqlx::query(
            r#"SELECT name, value FROM cookies WHERE domain = $1 ORDER BY name"#,
        )
        .bind(domain)
        .fetch_all(&*self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| (r.get(0), r.get(1))).collect())
    }

    /// Get all cookies across all domains.
    pub async fn get_all_cookies(&self) -> Result<Vec<(String, String, String)>, SqlxError> {
        let rows = sqlx::query(
            r#"SELECT domain, name, value FROM cookies ORDER BY domain, name"#,
        )
        .fetch_all(&*self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| (r.get(0), r.get(1), r.get(2))).collect())
    }

    /// Create a download job.
    pub async fn create_download_job(
        &self,
        url: &str,
        dest_path: &str,
        total_bytes: Option<i64>,
    ) -> Result<String, SqlxError> {
        let job_id = uuid::Uuid::new_v4().simple().to_string();

        sqlx::query(
            r#"
            INSERT INTO download_jobs (job_id, url, dest_path, total_bytes, status)
            VALUES ($1, $2, $3, $4, 'running')
            "#,
        )
        .bind(&job_id)
        .bind(url)
        .bind(dest_path)
        .bind(total_bytes)
        .execute(&*self.pool)
        .await?;

        Ok(job_id)
    }

    /// Get download job status.
    pub async fn get_download_job(
        &self,
        job_id: &str,
    ) -> Result<Option<(String, String, i64, i64, String, Option<String>)>, SqlxError> {
        let row = sqlx::query(
            r#"
            SELECT url, dest_path, COALESCE(total_bytes, 0), downloaded_bytes, status, error
            FROM download_jobs
            WHERE job_id = $1
            "#,
        )
        .bind(job_id)
        .fetch_optional(&*self.pool)
        .await?;

        Ok(row.map(|r| (
            r.get::<String, _>(0),
            r.get::<String, _>(1),
            r.get::<i64, _>(2),
            r.get::<i64, _>(3),
            r.get::<String, _>(4),
            r.get::<Option<String>, _>(5),
        )))
    }

    /// Create a new conversation and return its ID.
    pub async fn create_conversation(&self) -> Result<Uuid, SqlxError> {
        log::info!("Database: creating new conversation");
        let row = sqlx::query(
            r#"
            INSERT INTO conversations (id) VALUES ($1) RETURNING id
            "#,
        )
        .bind(Uuid::new_v4())
        .fetch_one(&*self.pool)
        .await?;

        let id = row.get::<Uuid, _>(0);
        log::info!("Database: created conversation id={}", id);
        Ok(id)
    }

    /// Save a message to a conversation.
    pub async fn save_message(
        &self,
        conversation_id: Uuid,
        role: &str,
        content: &str,
    ) -> Result<Uuid, SqlxError> {
        log::info!(
            "Database: saving message — conv={}, role={}, len={}",
            conversation_id, role, content.len()
        );
        let row = sqlx::query(
            r#"
            INSERT INTO conversation_messages (id, conversation_id, role, content)
            VALUES ($1, $2, $3, $4) RETURNING id
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(conversation_id)
        .bind(role)
        .bind(content)
        .fetch_one(&*self.pool)
        .await?;

        let id = row.get::<Uuid, _>(0);
        log::info!("Database: saved message id={} for conv={}", id, conversation_id);
        Ok(id)
    }

    /// Get all messages for a conversation.
    pub async fn get_messages(
        &self,
        conversation_id: Uuid,
    ) -> Result<Vec<ConversationMessage>, SqlxError> {
        log::info!("Database: fetching messages for conv={}", conversation_id);
        let rows = sqlx::query(
            r#"
            SELECT id, conversation_id, role, content, created_at
            FROM conversation_messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            "#,
        )
        .bind(conversation_id)
        .fetch_all(&*self.pool)
        .await?;

        let count = rows.len();
        log::info!("Database: fetched {} messages for conv={}", count, conversation_id);

        let messages = rows
            .into_iter()
            .map(|row| ConversationMessage {
                id: row.get(0),
                conversation_id: row.get(1),
                role: row.get(2),
                content: row.get(3),
                created_at: row.get(4),
            })
            .collect();

        Ok(messages)
    }

    /// Get all conversations.
    pub async fn get_conversations(&self) -> Result<Vec<Conversation>, SqlxError> {
        log::info!("Database: fetching all conversations");
        let rows = sqlx::query(
            r#"
            SELECT id, created_at
            FROM conversations
            ORDER BY created_at DESC
            "#,
        )
        .fetch_all(&*self.pool)
        .await?;

        let count = rows.len();
        log::info!("Database: fetched {} conversations", count);

        let conversations = rows
            .into_iter()
            .map(|row| Conversation {
                id: row.get(0),
                created_at: row.get(1),
            })
            .collect();

        Ok(conversations)
    }
}
