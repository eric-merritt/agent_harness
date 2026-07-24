# Agent Harness Architecture Plan

## Overview
A modular, event-driven agent harness built in Rust using Ratatui, following a hybrid architecture with an async event core and component-based modules.

---

## Phase 1: Foundation & Core Infrastructure (Priority: CRITICAL)

### Stage 1.1: Project Scaffolding & Event Core
**Goal:** Establish the async event backbone that all modules will plug into.

**Tasks:**
- Initialize Cargo workspace with modular crate structure
- Define core `AppEvent` enum (input, timer, module-specific, system events)
- Implement unified event bus using `tokio::sync::broadcast` + `mpsc` channels
- Create central event loop with task scheduler
- Set up shared `AppState` with `Arc<Mutex<>>` or `tokio::sync::RwLock`
- Implement zero-copy memory primitives using `zerocopy` crate
- Create memory controller abstraction for unified GPU+CPU RAM view

**Deliverables:**
- `harness-core/` crate with event types, state management, event bus
- `harness-memory/` crate with zerocopy wrappers and memory controller
- Basic event loop that processes events and updates state
- Unit tests for event routing and state mutations

**Dependencies:** tokio, zerocopy, parking_lot, thiserror

---

### Stage 1.2: Ratatui TUI Shell
**Goal:** Create the renderable UI framework with mouse support.

**Tasks:**
- Initialize crossterm backend with raw mode
- Implement main layout engine (vertical/horizontal splits)
- Create clickable tab bar component for toolbar navigation
- Add mouse event capture and conversion to `AppEvent::MouseClick`
- Build basic viewport/scrolling system for chat interface
- Implement render cycle triggered by state change flags
- Create theme/color configuration system

**Deliverables:**
- `harness-tui/` crate with widgets, layouts, input handling
- Working TUI with tabs, mouse clicks, and basic navigation
- Render loop integrated with event core
- Example views demonstrating layout composition

**Dependencies:** ratatui, crossterm, unicode-segmentation

---

### Stage 1.3: Module Trait & Plugin System
**Goal:** Define the contract all constraint modules must implement.

**Tasks:**
- Create `Module` trait with lifecycle methods: `init()`, `update()`, `view()`, `shutdown()`
- Define `ModuleMessage` enum for inter-module communication
- Implement module registry with dynamic loading capability
- Create module context struct with event sender, state reader, config access
- Build module error handling and recovery system
- Document module development guidelines

**Deliverables:**
- `harness-modules/` crate with trait definitions and registry
- Example "Hello World" module demonstrating the pattern
- Documentation for module authors
- Integration tests showing module isolation

---

## Phase 2: Core Agent Functionality (Priority: HIGH)

### Stage 2.1: Chat Interface Module
**Goal:** Implement the primary user-agent interaction view.

**Tasks:**
- Create `ChatState` with message history, input buffer, scroll position
- Build message rendering with syntax highlighting for code blocks
- Implement input field with multi-line support and history navigation
- Add user vs agent message differentiation (colors, icons)
- Create streaming response display for token-by-token rendering
- Integrate with event core for send/receive cycles
- Add conversation save/load functionality

**Deliverables:**
- `module-chat/` crate as standalone module
- Full chat interface with scrolling, input, and display
- Message persistence layer
- Demo with mock agent responses

---

### Stage 2.2: Tool System Redesign
**Goal:** Migrate 100+ tools to Rust with modular architecture.

**Tasks:**
- Define `Tool` trait with `name()`, `description()`, `parameters()`, `execute()`
- Create tool registry with capability-based discovery
- Implement tool parameter validation using `serde_json` schema
- Build tool execution sandbox with timeout and resource limits
- Create tool result caching layer
- Design tool composition system (pipeline multiple tools)
- Migrate high-priority tools first (file ops, shell, search, http)
- Create tool documentation generator

**Deliverables:**
- `harness-tools/` crate with trait, registry, and base implementations
- 20-30 core tools migrated and tested
- Tool testing framework with mock contexts
- Performance benchmarks for tool execution

**Dependencies:** serde, serde_json, schemars, tempfile

---

### Stage 2.3: MCP Connection Module
**Goal:** Enable Model Context Protocol connectivity.

**Tasks:**
- Implement MCP client with JSON-RPC transport
- Create connection pool for multiple MCP servers
- Build tool discovery from MCP capabilities
- Add resource subscription and update streaming
- Implement prompt template fetching from MCP servers
- Create fallback mechanisms for connection failures
- Add authentication and credential management

**Deliverables:**
- `module-mcp/` crate with full MCP client implementation
- Configuration system for MCP server endpoints
- Integration with tool registry (MCP tools appear alongside native)
- Connection health monitoring dashboard in TUI

**Dependencies:** jsonrpc-core, tokio-tungstenite, reqwest

---

## Phase 3: Advanced Agent Loops (Priority: HIGH)

### Stage 3.1: Graph-Based Tool Database
**Goal:** Create intelligent tool matching through agent planning.

**Tasks:**
- Design graph schema: nodes (tools, capabilities), edges (dependencies, prerequisites)
- Implement graph storage using `petgraph` or custom adjacency list
- Create "planning agent" prompt template for task decomposition
- Build task list parser to extract structured requirements
- Implement graph traversal algorithm for tool matching
- Add confidence scoring for tool-task matches
- Create visualization of tool graph in TUI (ASCII or simple diagram)
- Build cache for common task→tool mappings

**Deliverables:**
- `module-tool-graph/` crate with graph engine and matching logic
- Planning agent integration that produces structured task lists
- TUI view showing matched tools and reasoning
- Benchmark suite for matching accuracy

**Dependencies:** petgraph, serde_graph, regex

---

### Stage 3.2: Generative Adversarial Loop (GAN Loop)
**Goal:** Implement three-agent system with planning, generation, and evaluation.

**Tasks:**
- Create `GANLoopState` with task list, completion status, evaluator notes
- Implement Planning Agent module with task decomposition logic
- Build Generating Agent with clean-slate reinitialization per iteration
- Create Evaluating Agent with binary completion marking + note attachment
- Design iteration cycle: plan → generate → evaluate → repeat
- Add progress tracking and visualization (task completion percentage)
- Implement loop termination conditions (all complete, max iterations, timeout)
- Create checkpoint system to save/restore loop state
- Build conflict resolution for evaluator disagreements

**Deliverables:**
- `module-gan-loop/` crate with full three-agent orchestration
- TUI dashboard showing all three agents' current work
- Task list view with completion status and evaluator notes
- Configuration for iteration limits and timeouts

---

### Stage 3.3: Autoresearch Loop
**Goal:** Implement metric-driven iterative improvement system.

**Tasks:**
- Define `Metric` trait with `measure()`, `compare()`, `target()`
- Create branch management system using `git2` (feature branch per iteration)
- Implement test runner with metric collection
- Build change validation: keep if improves, discard if worse/same
- Add time-boxed iterations (5-15 min configurable)
- Create metric dashboard with historical trend visualization
- Implement iteration limit and manual stop controls
- Build rollback mechanism for bad merges
- Add statistical significance testing for metric changes
- Create research report generator summarizing iterations

**Deliverables:**
- `module-autoresearch/` crate with full loop implementation
- Git integration for branch management and merging
- Metric collection and comparison engine
- TUI view showing current iteration, metric trends, and decision log
- Configuration presets for different metric types (token/sec, iter/sec, etc.)

**Dependencies:** git2, statrs, indicatif

---

## Phase 4: The Agent Smith Workshop (Priority: MEDIUM)

### Stage 4.1: Component Hopper & Anvil Interface
**Goal:** Build visual workshop for assembling agent configurations.

**Tasks:**
- Create component library browser (hopper) with categorized blocks
- Implement drag-and-drop or click-to-add component selection
- Build "anvil" canvas for arranging components into agent designs
- Add component property editors (prompts, configs, connections)
- Create component linking system for multi-agent loops
- Implement save/load for agent blueprints (YAML/JSON)
- Add blueprint validation before deployment
- Build preview mode to test blueprint behavior
- Create template library for common agent patterns

**Deliverables:**
- `module-smith/` crate with full workshop UI
- Visual component editor with hopper and anvil views
- Blueprint serialization and validation
- Template gallery with example agent designs
- Export functionality to generate runnable agent configs

---

### Stage 4.2: Dynamic Component Loading
**Goal:** Enable runtime addition of new components without recompilation.

**Tasks:**
- Implement dynamic library loading using `libloading`
- Create component ABI specification for plugin compatibility
- Build hot-reload system for component updates
- Add component versioning and dependency checking
- Create sandbox for untrusted components
- Implement component marketplace interface (local or remote)
- Add component signing and verification

**Deliverables:**
- Dynamic loading infrastructure in `harness-modules`
- Example plugins loaded at runtime
- Security documentation for component sourcing
- TUI indicator for loaded/unloaded components

**Dependencies:** libloading, semver

---

## Phase 5: Memory & Performance Optimization (Priority: MEDIUM-HIGH)

### Stage 5.1: Zero-Copy Memory Implementation
**Goal:** Maximize performance through zero-copy data handling.

**Tasks:**
- Audit all data paths for copy opportunities
- Implement `zerocopy` traits for all serializable structures
- Create arena allocators for frequent short-lived allocations
- Build shared memory regions for inter-module communication
- Add memory profiling and leak detection
- Optimize event payload sizes with zero-copy serialization
- Implement memory pools for reusable buffers
- Create benchmark suite comparing copy vs zero-copy paths

**Deliverables:**
- Comprehensive zerocopy usage across all crates
- Memory profiling dashboard in TUI
- Performance benchmarks showing improvements
- Documentation on zero-copy best practices for module authors

---

### Stage 5.2: Unified Memory Controller
**Goal:** Abstract GPU + CPU RAM into single addressable space.

**Tasks:**
- Design memory controller API with unified allocation interface
- Implement CPU memory backend (standard heap)
- Add GPU memory backend using `wgpu` or `cuda-rs`
- Create automatic tiering: hot data in GPU, cold in CPU
- Build page migration system between tiers
- Add memory pressure handling and eviction policies
- Implement memory statistics and monitoring
- Create NUMA-aware allocation for multi-GPU systems
- Add persistent memory support if available (PMEM)

**Deliverables:**
- `harness-memory-controller/` crate with unified API
- Automatic data movement between CPU/GPU
- TUI memory dashboard showing utilization across tiers
- Configuration for tiering policies and thresholds
- Benchmarks demonstrating performance gains

**Dependencies:** wgpu or cudarc, memmap2

---

## Phase 6: Polish & Production Readiness (Priority: LOW-MEDIUM)

### Stage 6.1: Enhanced Mouse Interactivity
**Goal:** Make full TUI mouse-navigable.

**Tasks:**
- Add hover states and visual feedback for clickable elements
- Implement right-click context menus throughout
- Create resizable panes with mouse drag
- Add scroll wheel support with momentum
- Build tooltip system for discoverability
- Implement double-click actions (select word, open file)
- Add keyboard shortcuts overlay accessible via mouse

**Deliverables:**
- Fully mouse-navigable interface
- Context-sensitive cursors and hover effects
- Accessibility improvements for mouse-only users

---

### Stage 6.2: Configuration & Persistence
**Goal:** Robust configuration management and state persistence.

**Tasks:**
- Create hierarchical config system (default → user → project → session)
- Implement config hot-reload without restart
- Add session restore on crash or unexpected exit
- Build export/import for entire harness state
- Create config validation with helpful error messages
- Add environment variable overrides
- Implement secrets management for API keys

**Deliverables:**
- `harness-config/` crate with full config management
- TUI config editor with validation
- Session recovery functionality
- Documentation for configuration options

**Dependencies:** serde, toml, directories

---

### Stage 6.3: Testing & Documentation
**Goal:** Production-quality test coverage and docs.

**Tasks:**
- Achieve 80%+ unit test coverage on core crates
- Create integration test suite for full workflows
- Build end-to-end tests with simulated agents
- Write comprehensive API documentation
- Create user guide with tutorials
- Record demo videos for key features
- Set up CI/CD pipeline with automated testing
- Create contribution guidelines for new modules

**Deliverables:**
- Full test suite passing in CI
- Published rustdoc documentation
- User manual and quickstart guide
- Example projects demonstrating each feature

---

## Phase 7: Extended Capabilities (Priority: LOW - Future)

### Stage 7.1: Distributed Agent Execution
- Multi-machine agent coordination
- Work stealing between harness instances
- Centralized orchestration dashboard

### Stage 7.2: Advanced Analytics
- Agent performance profiling
- Cost tracking across LLM providers
- A/B testing framework for agent strategies

### Stage 7.3: Ecosystem Integration
- VSCode extension for harness control
- CLI tool for headless operation
- Web UI alternative to TUI
- API server for remote harness access

---

## Dependency Graph & Parallelization

```
Phase 1 (Sequential): 1.1 → 1.2 → 1.3
                      │
Phase 2 (Parallel):   ├── 2.1 (Chat)
                      ├── 2.2 (Tools) ──→ 2.3 (MCP)
                      │
Phase 3 (Parallel):   ├── 3.1 (Tool Graph)
                      ├── 3.2 (GAN Loop)
                      └── 3.3 (Autoresearch)
                      │
Phase 4 (Sequential): 4.1 → 4.2
                      │
Phase 5 (Parallel):   ├── 5.1 (Zero Copy)
                      └── 5.2 (Memory Controller)
                      │
Phase 6 (Parallel):   ├── 6.1 (Mouse)
                      ├── 6.2 (Config)
                      └── 6.3 (Testing)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Complexity creep | Strict module boundaries, MVP-first approach |
| Performance issues | Early profiling, zerocopy from day one |
| Module coupling | Event-only communication, no direct calls |
| Ratatui limitations | Fallback to terminal graphics, custom widgets |
| Tool migration scope | Prioritize top 20 tools, lazy migrate rest |
| GAN loop instability | Extensive testing, human-in-the-loop override |
| Memory controller bugs | Extensive property testing, fuzzing |

---

## Success Metrics

- **Modularity**: Any module can be removed without compilation errors
- **Performance**: <16ms frame time at 60fps with 10k messages/sec
- **Memory**: <10% overhead from abstraction layers
- **Developer Experience**: New module created and integrated in <1 hour
- **User Experience**: All features accessible via mouse or keyboard
- **Reliability**: 99.9% uptime in 24hr stress tests

---

## Next Steps

1. **Immediate**: Start Phase 1.1 with project scaffolding
2. **Week 1**: Complete event core and basic TUI shell
3. **Week 2-3**: Implement module system and chat interface
4. **Week 4-6**: Migrate critical tools and add MCP support
5. **Month 2**: Build GAN and Autoresearch loops
6. **Month 3**: Complete Agent Smith workshop
7. **Month 4**: Optimize memory and polish UX

This plan provides a clear, phased approach to building your modular agent harness while maintaining flexibility for iteration and adjustment based on learnings during implementation.
