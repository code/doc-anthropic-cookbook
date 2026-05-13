import Anthropic from "@anthropic-ai/sdk";
import { LinearClient } from "@linear/sdk";
import { getAccessToken } from "./oauth";

const anthropic = new Anthropic();

const CLAUDE_AGENT_ID = process.env.CLAUDE_AGENT_ID!;
const CLAUDE_ENVIRONMENT_ID = process.env.CLAUDE_ENVIRONMENT_ID!;

interface AgentSessionEvent {
  action: string;
  agentSession: {
    id: string;
    issue?: { identifier: string; title: string; description?: string | null } | null;
    comment?: { body: string } | null;
  };
  agentActivity?: { content: { body?: string } } | null;
  organizationId: string;
  promptContext?: string | null;
  previousComments?: Array<{ body: string }> | null;
}

// Fire-and-forget: create the CMA session, attach routing metadata, send the
// prompt, return. The reply path is handled in cma-webhook.ts when Anthropic
// POSTs `session.status_idled`.
export async function kickoffAgentSession(event: AgentSessionEvent) {
  const { agentSession, organizationId } = event;

  const accessToken = await getAccessToken(organizationId);
  const linear = new LinearClient({ accessToken });

  // Linear requires a first activity within 10s.
  await linear.createAgentActivity({
    agentSessionId: agentSession.id,
    content: { type: "thought", body: "Thinking..." },
  });

  // Stash the Linear routing info on the CMA session. The idle webhook later
  // delivers only a session ID; we read this metadata back to know where to
  // post the reply.
  const session = await anthropic.beta.sessions.create({
    agent: CLAUDE_AGENT_ID,
    environment_id: CLAUDE_ENVIRONMENT_ID,
    metadata: {
      linear_session_id: agentSession.id,
      linear_org_id: organizationId,
    },
  });

  await anthropic.beta.sessions.events.send(session.id, {
    events: [{ type: "user.message", content: [{ type: "text", text: buildPrompt(event) }] }],
  });

  console.log(`[agent] kickoff linear=${agentSession.id} claude=${session.id}`);
}

function buildPrompt(event: AgentSessionEvent): string {
  if (event.promptContext) return event.promptContext;

  const parts: string[] = [];
  const { agentSession, agentActivity, previousComments } = event;

  if (agentSession.issue) {
    parts.push(`Issue: ${agentSession.issue.identifier} - ${agentSession.issue.title}`);
    if (agentSession.issue.description) parts.push(`Description: ${agentSession.issue.description}`);
  }
  if (previousComments?.length) {
    parts.push("Previous comments:\n" + previousComments.map((c) => `- ${c.body}`).join("\n"));
  }
  const msg = agentActivity?.content?.body ?? agentSession.comment?.body;
  if (msg) parts.push(`User message: ${msg}`);

  return parts.join("\n\n") || "Hello! How can I help?";
}
