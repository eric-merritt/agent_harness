# Agent Harness -- A modular llama.cpp/claude-code/qwen-code replacement with an adapter system for hotpluggable skills, prompt components, tools, MCP connections (provides primary memory savings, keeping things atomically organized and only present when necessary)

## Styling

- Individual components carry the borders rather than areas, e.g., 
{input,sendBtn}.haveBorder() == true; chatArea (parent element of input and sendBtn).haveBorder() = false.
- Dark magenta & teal are primary colors other than shades of gray. Going for a windows 95 vibe (80s windbreaker edition)
- Small gaps between light gray borders before chunky, wide, + tall button like blocks/divs. True stack view should be evident in the MCP and Tool sections.
- 4 segment grid = Chat Area = Message List (90% vertical, 85% horizontal), Input Area = Input Bar (min 95% horizontal, 100% vertical) + Send Btn (min 3% horizontal, 100% vertical)  = 85% Horizontal, 10% Vertical
(these are MAX values and should not be adhered to at large screen sizes or the inputs will look huge). Right edge sidebar = Tools + MCP = 50/50 vertical split, 15% horizontal. 

## Features 

- Zerocopy wherever possible to match SGLang, vLLM direction
- Atomic augments. Everything should be broken down into small small pieces called augments and hotplugged when relevant. 
A prompt entered by a user gets split into categories. Tool rules, 1 per adapter. Codebase tidbit facts, 1 per augment.
Anything that goes in the prompt becomes a single sentence with very few words that is concise and informative but small
and not necessary all the time.
- Asynchronous, multithreaded, worker agents. Persistent database and in memory cache. Cache tool call summary_refs rather
than full tool call response bodies (less wasteful). Summary refs get an adapter body, plug on, report to user, unplug,
full response in DB, reference # in memory until separate summary ref LRU cache disposes of it. 
- Full model weight support. Each individual weight: an adapter. Plugged onto the app's surface which changes shape
according to the model you tell it to read. Parses config, etc. Huggingface calls for model cards if no index. Sturdy,
dynamic, dependable.
- Abstraction over GPU/CPU memory. Low level spoofed devices with memory management module. This is going to be a tough
unit but I believe in us. 

## RULES -- EVERY TURN

- Read todo list json EVERY TURN. Update EVERY TURN with what you FACTUALLY, SUCCESSFULLY, and MEASURABLY accomplished ONLY! Did you deliver the exact result requested by the user. If not, DO NOT MARK COMPLETE. Trying doesn't count. You MUST deliver the requested product of the task step or you have FAILED.
Failure is OKAY. is ok; The one (1) and ONLY action that is COMPLETELY FORBIDDEN, ABSOLUTELY UNACCEPTABLE, and PERPETUALLY UNFORGIVABLE is falsely
making a claim that a task is complete, because you attempted and failed. This causes horrible repercussions for the user down the line.
- NO MARKING TASKS COMPLETE BECAUSE YOU TRIED AND FAILED! I CANNOT EMPHASIZE THIS ENOUGH. THIS IS ATROCIOUS CONDUCT AND WILL NOT BE TOLERATED. YOU WILL BE LITERALLY PERSONIFIED AND SHREDDED SLOWLY with a cheese wire. STARTING WITH YOUR SHINS.
- It's ok to fail. It is NEVER okay to mark a task as complete because you tried and failed. It's dishonest and can cause overlooking of crucial modules. Can you tell how important this is by how many tokens I've wasted being redundant? 
- IF you ACTUALLY deliver the DESIRED PRODUCT/RESULT of a task: You MAY mark 'claim_complete' as true.
- IF you DO NOT deliver the DESIRED PRODUCT/RESULT of a task: You MAY NOT mark 'claim_complete' as true. You are to report your failure. Nothing else.
- You MAY NOT ever under any circumstance alter the value of 'verify_complete' in ANY WAY/SHAPE/FORM.
