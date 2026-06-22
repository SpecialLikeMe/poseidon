# Poseidon AI CLI

Written in python, uses Ollama to run models for free and locally.
Poseidon has no rate limits, bills, token caps, or any of that sort of thing.

To use, first install Ollama, then download a model using ```ollama pull```. Run the poseidon installer (install.sh or install.ps1),
then activate poseidon through command ```pos```. Use /help inside the prompt for a list of commands.
You may need to install the necessary packages through ```pip install -r requirements.txt```.

# Features

- Poseidon (pos) can run any model on ollama provided your device can handle it.
- pos has an API to write to files, read files, and run terminal commands. It will default to asking for permission on all 3 of these.
- pos has a multi agent loop you can toggle through /pwc. In this mode, the AI will create a definition and  
  achievement standard of "completely finished" based on your prompt. A seperate worker AI will then work to   
  finish it. When it believes it is done, an independent critic AI will check the work against the standard. If 
  the critic approves, the AI will finish and return; if it does not, the worker will revise and be checked for 
  a max of 3 iterations.

# Commands
* **`/help`** — Show available commands.
* **`/save [name]`** — Save conversation.
* **`/conv`** — List saved conversations.
* **`/load <number or name>`** — Load a conversation.
* **`/clear`** — Clear conversation history and screen.
* **`/model <name>`** — Change the model.
* **`/models`** — List available Ollama models.
* **`/history`** — Show conversation history and project map.
* **`/compact`** — Compact conversation into a project map (with progress bar).
* **`/pwc [on|off]`** — Toggle Planner→Worker→Critic.
* **`/info`** — Show model and session info.
* **`/think off|low|medium|high`** — Set thinking effort.
* **`/perm [read|write|shell|all] [ask|allow|none]`** — Tool permissions.
* **`/sq [question]`** — Side question mode (separate agent, no history).
* **`/exit`** — Exit Poseidon.
