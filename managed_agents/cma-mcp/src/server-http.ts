#!/usr/bin/env bun
// Streamable HTTP transport — for claude.ai web Connectors (or any remote MCP client).
// Web-standard Request/Response, so this same file runs on Bun, Node 18+, Deno,
// and (with env plumbing) Cloudflare Workers.
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { timingSafeEqual } from "node:crypto";
import { registerTools } from "./tools";

const TOKEN = process.env.CMA_MCP_TOKEN;
if (!TOKEN) throw new Error("CMA_MCP_TOKEN is required for the HTTP server");
const PORT = Number(process.env.PORT) || 3000;

// Stateless mode (no sessionIdGenerator): each HTTP request gets its own
// McpServer + transport. Fine because the tools themselves are stateless —
// Claude holds the CMA session_id across turns.
function newTransport() {
  const server = new McpServer({ name: "cma-mcp", version: "0.1.0" });
  registerTools(server);
  const transport = new WebStandardStreamableHTTPServerTransport({});
  server.connect(transport);
  return transport;
}

function authorized(req: Request): boolean {
  const provided = req.headers.get("authorization") ?? "";
  const expected = `Bearer ${TOKEN}`;
  const a = Buffer.from(provided);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/health" && req.method === "GET") {
      return new Response("ok");
    }
    if (url.pathname !== "/mcp") {
      return new Response("Not Found", { status: 404 });
    }
    // The only thing standing between the internet and your ANTHROPIC_API_KEY's
    // CMA quota. Do not remove.
    if (!authorized(req)) {
      return new Response("Unauthorized", { status: 401 });
    }

    return newTransport().handleRequest(req);
  },
});

console.error(`[cma-mcp] HTTP server on :${PORT}/mcp`);
