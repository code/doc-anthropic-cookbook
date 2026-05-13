import { LinearWebhookClient } from "@linear/sdk/webhooks";
import { handleOAuthAuthorize, handleOAuthCallback } from "./oauth";
import { kickoffAgentSession } from "./agent";
import { handleCmaWebhook } from "./cma-webhook";

const PORT = Number(process.env.PORT) || 3000;
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`;

for (const v of [
  "LINEAR_WEBHOOK_SIGNING_SECRET",
  "ANTHROPIC_WEBHOOK_SIGNING_KEY",
  "CLAUDE_AGENT_ID",
  "CLAUDE_ENVIRONMENT_ID",
]) {
  if (!process.env[v]) {
    console.error(`FATAL: ${v} is required`);
    process.exit(1);
  }
}

const linearHandler = new LinearWebhookClient(
  process.env.LINEAR_WEBHOOK_SIGNING_SECRET!,
).createHandler();

linearHandler.on("AgentSessionEvent", (event) => {
  console.log(`[linear] ${event.action} session=${event.agentSession.id}`);
  kickoffAgentSession(event).catch((err) =>
    console.error("[linear] kickoff error:", err),
  );
});

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/" && req.method === "GET") {
      return Response.json({ status: "ok" });
    }
    if (url.pathname === "/oauth/authorize" && req.method === "GET") {
      return handleOAuthAuthorize();
    }
    if (url.pathname === "/oauth/callback" && req.method === "GET") {
      return handleOAuthCallback(url);
    }
    if (url.pathname === "/linear-webhook" && req.method === "POST") {
      return linearHandler(req);
    }
    if (url.pathname === "/cma-webhook" && req.method === "POST") {
      return handleCmaWebhook(req);
    }
    return new Response("Not Found", { status: 404 });
  },
});

console.log(`Bridge running at ${BASE_URL}`);
console.log(`  Install agent:  ${BASE_URL}/oauth/authorize`);
console.log(`  Linear webhook: ${BASE_URL}/linear-webhook`);
console.log(`  CMA webhook:    ${BASE_URL}/cma-webhook`);
