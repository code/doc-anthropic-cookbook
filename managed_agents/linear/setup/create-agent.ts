// One-time: create the Claude agent + environment. Copy the printed IDs into .env.local.
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

const env = await anthropic.beta.environments.create({
  name: `linear-bridge-${Date.now()}`,
  config: { type: "cloud", networking: { type: "unrestricted" } },
});

const agent = await anthropic.beta.agents.create({
  name: "Linear Assistant",
  model: "claude-opus-4-7",
  system:
    "You are a helpful assistant embedded in Linear. Keep replies concise and actionable — they are posted as comments. Do not invent issue IDs, users, or project names.",
  tools: [{ type: "agent_toolset_20260401", default_config: { enabled: true } }],
});

console.log("\nAdd to .env.local:");
console.log(`CLAUDE_ENVIRONMENT_ID=${env.id}`);
console.log(`CLAUDE_AGENT_ID=${agent.id}`);
