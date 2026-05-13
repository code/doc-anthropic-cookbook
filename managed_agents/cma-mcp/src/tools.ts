import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import * as cma from "./cma";

function json(result: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
}

/** Register all nine CMA tools on an MCP server. Shared by stdio + HTTP entrypoints. */
export function registerTools(server: McpServer) {
  // --- Tier 1: 1:1 session primitives -------------------------------------

  server.tool(
    "list_agents",
    "List Managed Agents available in this workspace. Use name_contains to filter in busy workspaces.",
    {
      limit: z.number().int().min(1).max(200).optional().describe("Default 50"),
      name_contains: z.string().optional().describe("Case-insensitive substring match on agent name"),
    },
    async ({ limit, name_contains }) => json(await cma.listAgents({ limit, name_contains })),
  );

  server.tool(
    "get_agent",
    "Get the full configuration of one Managed Agent.",
    { agent_id: z.string() },
    async ({ agent_id }) => json(await cma.getAgent(agent_id)),
  );

  server.tool(
    "create_session",
    "Start a new session with a Managed Agent. Returns a session_id — pass it to send_message / wait_for_idle on every subsequent turn.",
    {
      agent_id: z.string().describe("Agent ID from list_agents"),
      title: z.string().optional(),
    },
    async ({ agent_id, title }) => json(await cma.createSession(agent_id, title)),
  );

  server.tool(
    "send_message",
    "Send a user message to a running session. Returns immediately; call wait_for_idle to get the reply.",
    {
      session_id: z.string(),
      text: z.string().describe("The user's message, verbatim"),
    },
    async ({ session_id, text }) => json(await cma.sendMessage(session_id, text)),
  );

  server.tool(
    "interrupt",
    "Interrupt a running session (agent stops at the next safe point and goes idle).",
    { session_id: z.string() },
    async ({ session_id }) => json(await cma.interrupt(session_id)),
  );

  server.tool(
    "get_session",
    "Get a session's current status, usage, and metadata.",
    { session_id: z.string() },
    async ({ session_id }) => json(await cma.getSession(session_id)),
  );

  server.tool(
    "list_events",
    "List a session's event log (messages, tool calls, status changes). Use after_id to fetch only new events.",
    {
      session_id: z.string(),
      after_id: z.string().optional().describe("Only return events after this event ID"),
    },
    async ({ session_id, after_id }) => json(await cma.listEvents(session_id, after_id)),
  );

  server.tool(
    "archive_session",
    "Archive a session when the conversation is finished. Frees resources; the session becomes read-only.",
    { session_id: z.string() },
    async ({ session_id }) => json(await cma.archiveSession(session_id)),
  );

  // --- Tier 1.5: the SSE→request/response shim ----------------------------

  server.tool(
    "wait_for_idle",
    "Block until the agent finishes the current turn (session goes idle) and return its reply text. Call this immediately after send_message. Typical loop per user turn: send_message → wait_for_idle.",
    {
      session_id: z.string(),
      timeout_sec: z.number().int().min(5).max(600).optional().describe("Default 120"),
    },
    async ({ session_id, timeout_sec }) =>
      json(await cma.waitForIdle(session_id, timeout_sec)),
  );
}
