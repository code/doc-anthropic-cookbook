#!/usr/bin/env bun
// stdio transport — for Claude Desktop / Claude Code.
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { registerTools } from "./tools";

const server = new McpServer({ name: "cma-mcp", version: "0.1.0" });
registerTools(server);

await server.connect(new StdioServerTransport());
console.error("[cma-mcp] stdio server ready");
