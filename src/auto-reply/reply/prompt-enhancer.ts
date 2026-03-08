import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { isCliProvider } from "../../agents/model-selection.js";
import { runEmbeddedPiAgent } from "../../agents/pi-embedded.js";
import type { OpenClawConfig } from "../../config/config.js";
import {
  updateSessionStore,
  type PromptEnhancerDraft,
  type SessionEntry,
} from "../../config/sessions.js";
import type { MsgContext, TemplateContext } from "../templating.js";
import type { GetReplyOptions, ReplyPayload } from "../types.js";
import type { CommandContext } from "./commands-types.js";

const PROMPT_COMMAND = "/prompt";
const PROMPT_DRAFT_TTL_MS = 30 * 60 * 1000;
const PROMPT_ENHANCER_TIMEOUT_MS = 30_000;
const MAX_ENHANCED_PROMPT_CHARS = 8_000;
const MAX_LIST_ITEMS = 6;

const PROMPT_ENHANCER_SYSTEM_PROMPT = [
  "You are OpenClaw's prompt enhancement stage.",
  "Rewrite the latest user request into a clearer execution prompt for the main agent.",
  "Preserve the user's actual intent. Do not invent requirements, file names, tools, or constraints.",
  "Use prior transcript context only when it clearly clarifies the current request. Do not revive stale tasks.",
  "If important details are missing, add them to clarifyingQuestions instead of guessing.",
  "Keep enhancedPrompt ready for direct execution by the next agent turn.",
  "Prefer the same language as the user's latest message.",
  "Return strict JSON only with this exact schema:",
  '{"goal":"string","constraints":["string"],"assumptions":["string"],"clarifyingQuestions":["string"],"enhancedPrompt":"string"}',
].join("\n");

type PromptEnhancerModelOutput = {
  goal?: string;
  constraints?: string[];
  assumptions?: string[];
  clarifyingQuestions?: string[];
  enhancedPrompt?: string;
};

export type PromptEnhancerResult = { kind: "reply"; reply: ReplyPayload } | { kind: "continue" };

type PromptEnhancerParams = {
  ctx: MsgContext;
  sessionCtx: TemplateContext;
  cfg: OpenClawConfig;
  agentId: string;
  agentDir?: string;
  workspaceDir: string;
  sessionEntry?: SessionEntry;
  sessionStore?: Record<string, SessionEntry>;
  sessionKey: string;
  storePath?: string;
  command: CommandContext;
  cleanedBody: string;
  provider: string;
  model: string;
  opts?: GetReplyOptions;
};

function isPromptEnhancerEnabled(cfg: OpenClawConfig): boolean {
  return cfg.agents?.defaults?.promptEnhancer?.enabled === true;
}

function normalizeStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const normalized = value.map((entry) => String(entry).trim()).filter(Boolean);
  return normalized.length > 0 ? normalized.slice(0, MAX_LIST_ITEMS) : undefined;
}

function normalizeDraftText(value: unknown): string | undefined {
  const trimmed = typeof value === "string" ? value.trim() : "";
  return trimmed ? trimmed : undefined;
}

function clampPromptText(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return trimmed;
  }
  return trimmed.length > MAX_ENHANCED_PROMPT_CHARS
    ? `${trimmed.slice(0, MAX_ENHANCED_PROMPT_CHARS).trimEnd()}\n\n[Truncated by prompt enhancer]`
    : trimmed;
}

function parsePromptCommand(
  raw: string,
): { action: "show" | "run" | "cancel" | "edit"; args?: string } | null {
  const trimmed = raw.trim();
  if (!trimmed.toLowerCase().startsWith(PROMPT_COMMAND)) {
    return null;
  }
  const rest = trimmed.slice(PROMPT_COMMAND.length).trim();
  if (!rest) {
    return { action: "show" };
  }
  const [verb, ...args] = rest.split(/\s+/);
  const action = verb.trim().toLowerCase();
  if (action === "show" || action === "run" || action === "cancel") {
    return { action };
  }
  if (action === "edit") {
    const nextArgs = args.join(" ").trim();
    return { action: "edit", args: nextArgs || undefined };
  }
  return null;
}

function parseEnhancerJson(raw: string): PromptEnhancerModelOutput | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }
  const candidates = new Set<string>([trimmed]);
  const fencedMatch =
    trimmed.match(/```json\s*([\s\S]*?)\s*```/i) ?? trimmed.match(/```\s*([\s\S]*?)\s*```/);
  if (fencedMatch?.[1]) {
    candidates.add(fencedMatch[1].trim());
  }
  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    candidates.add(trimmed.slice(firstBrace, lastBrace + 1));
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        continue;
      }
      return parsed as PromptEnhancerModelOutput;
    } catch {
      continue;
    }
  }
  return null;
}

function toPromptDraft(params: {
  output: PromptEnhancerModelOutput;
  originalPrompt: string;
  provider: string;
  model: string;
}): PromptEnhancerDraft | null {
  const enhancedPrompt = normalizeDraftText(params.output.enhancedPrompt);
  if (!enhancedPrompt) {
    return null;
  }
  return {
    id: crypto.randomUUID(),
    createdAt: Date.now(),
    originalPrompt: params.originalPrompt.trim(),
    enhancedPrompt: clampPromptText(enhancedPrompt),
    goal: normalizeDraftText(params.output.goal),
    constraints: normalizeStringArray(params.output.constraints),
    assumptions: normalizeStringArray(params.output.assumptions),
    clarifyingQuestions: normalizeStringArray(params.output.clarifyingQuestions),
    modelProvider: params.provider,
    model: params.model,
  };
}

function buildCreatePrompt(originalPrompt: string): string {
  return [
    "Create a reviewed execution draft for the latest user request.",
    "Fill every field in the JSON schema. Use empty arrays when nothing applies.",
    "",
    "Latest user request:",
    originalPrompt.trim(),
  ].join("\n");
}

function buildEditPrompt(draft: PromptEnhancerDraft, feedback: string): string {
  return [
    "Revise the existing prompt draft using the latest user follow-up.",
    "Keep the same JSON schema. Preserve confirmed intent, incorporate the new feedback, and do not invent missing details.",
    "",
    "Original request:",
    draft.originalPrompt,
    "",
    "Current prompt draft:",
    draft.enhancedPrompt,
    "",
    "Latest user follow-up:",
    feedback.trim(),
  ].join("\n");
}

function formatSection(title: string, lines?: string[]): string | null {
  if (!lines || lines.length === 0) {
    return null;
  }
  return `${title}:\n${lines.map((line) => `- ${line}`).join("\n")}`;
}

export function buildPromptDraftReply(
  draft: PromptEnhancerDraft,
  mode: "new" | "updated",
): ReplyPayload {
  const sections = [
    mode === "new" ? "Prompt draft ready." : "Prompt draft updated.",
    draft.goal ? `Goal:\n${draft.goal}` : null,
    formatSection("Constraints", draft.constraints),
    formatSection("Assumptions", draft.assumptions),
    formatSection("Clarifying questions", draft.clarifyingQuestions),
    ["Enhanced prompt:", "```text", draft.enhancedPrompt, "```"].join("\n"),
    "Reply with `/prompt run`, `/prompt edit <changes>`, `/prompt show`, or `/prompt cancel`.",
  ].filter((value): value is string => Boolean(value));
  return { text: sections.join("\n\n") };
}

function buildPromptDraftMissingReply(): ReplyPayload {
  return {
    text: "No pending prompt draft. Send a normal message first to generate one.",
  };
}

function buildPendingDraftReminderReply(): ReplyPayload {
  return {
    text: "A prompt draft is waiting for review. Reply with `/prompt run`, `/prompt edit <changes>`, `/prompt show`, or `/prompt cancel`.",
  };
}

function buildPromptEnhancerErrorReply(message: string): ReplyPayload {
  return {
    text: `Prompt enhancement failed: ${message}`,
    isError: true,
  };
}

function resolveLatestPromptText(sessionCtx: TemplateContext, cleanedBody: string): string {
  const cleaned = cleanedBody.trim();
  if (cleaned) {
    return cleaned;
  }
  const stripped = sessionCtx.BodyStripped?.trim();
  if (stripped) {
    return stripped;
  }
  return sessionCtx.Body?.trim() ?? "";
}

function hasInboundMedia(sessionCtx: TemplateContext): boolean {
  return Boolean(
    sessionCtx.MediaPath || (sessionCtx.MediaPaths && sessionCtx.MediaPaths.length > 0),
  );
}

async function persistPromptDraft(params: {
  sessionEntry?: SessionEntry;
  sessionStore?: Record<string, SessionEntry>;
  sessionKey: string;
  storePath?: string;
  nextDraft?: PromptEnhancerDraft;
}): Promise<void> {
  if (!params.sessionStore) {
    return;
  }
  const current = params.sessionEntry ??
    params.sessionStore[params.sessionKey] ?? {
      sessionId: crypto.randomUUID(),
      updatedAt: Date.now(),
    };
  const next: SessionEntry = {
    ...current,
    updatedAt: Date.now(),
  };
  if (params.nextDraft) {
    next.promptEnhancerDraft = params.nextDraft;
  } else {
    delete next.promptEnhancerDraft;
  }
  params.sessionStore[params.sessionKey] = next;
  if (params.sessionEntry) {
    Object.assign(params.sessionEntry, next);
    if (!params.nextDraft) {
      delete params.sessionEntry.promptEnhancerDraft;
    }
  }
  if (params.storePath) {
    await updateSessionStore(params.storePath, (store) => {
      store[params.sessionKey] = next;
    });
  }
}

async function resolveActiveDraft(params: {
  sessionEntry?: SessionEntry;
  sessionStore?: Record<string, SessionEntry>;
  sessionKey: string;
  storePath?: string;
}): Promise<PromptEnhancerDraft | undefined> {
  const draft =
    params.sessionEntry?.promptEnhancerDraft ??
    params.sessionStore?.[params.sessionKey]?.promptEnhancerDraft;
  if (!draft) {
    return undefined;
  }
  if (Date.now() - draft.createdAt <= PROMPT_DRAFT_TTL_MS) {
    return draft;
  }
  await persistPromptDraft({
    sessionEntry: params.sessionEntry,
    sessionStore: params.sessionStore,
    sessionKey: params.sessionKey,
    storePath: params.storePath,
    nextDraft: undefined,
  });
  return undefined;
}

async function withEnhancerSessionFile<T>(
  baseSessionFile: string | undefined,
  fn: (sessionFile: string, sessionId: string) => Promise<T>,
): Promise<T> {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-prompt-enhancer-"));
  const tempSessionFile = path.join(tempDir, "session.jsonl");
  const sessionId = `prompt-enhancer-${crypto.randomUUID()}`;
  try {
    if (baseSessionFile?.trim()) {
      await fs.copyFile(path.resolve(baseSessionFile), tempSessionFile).catch(async () => {
        await fs.writeFile(tempSessionFile, "", "utf-8");
      });
    } else {
      await fs.writeFile(tempSessionFile, "", "utf-8");
    }
    return await fn(tempSessionFile, sessionId);
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
  }
}

function extractEnhancerReplyText(payloads?: Array<{ text?: string }>): string {
  return (
    payloads
      ?.map((payload) => payload.text?.trim())
      .filter((value): value is string => Boolean(value))
      .join("\n")
      .trim() ?? ""
  );
}

async function runPromptEnhancerModel(params: {
  mode: "create" | "edit";
  promptText: string;
  draft?: PromptEnhancerDraft;
  provider: string;
  model: string;
  agentId: string;
  agentDir?: string;
  workspaceDir: string;
  cfg: OpenClawConfig;
  sessionEntry?: SessionEntry;
  sessionKey: string;
}): Promise<PromptEnhancerDraft> {
  const taskPrompt =
    params.mode === "edit" && params.draft
      ? buildEditPrompt(params.draft, params.promptText)
      : buildCreatePrompt(params.promptText);
  const originalPrompt =
    params.mode === "edit" && params.draft ? params.draft.originalPrompt : params.promptText.trim();

  const modelOutput = await withEnhancerSessionFile(
    params.sessionEntry?.sessionFile,
    async (sessionFile, sessionId) => {
      const result = await runEmbeddedPiAgent({
        sessionId,
        sessionFile,
        agentId: params.agentId,
        workspaceDir: params.workspaceDir,
        agentDir: params.agentDir,
        config: params.cfg,
        prompt: taskPrompt,
        provider: params.provider,
        model: params.model,
        disableTools: true,
        thinkLevel: "off",
        reasoningLevel: "off",
        verboseLevel: "off",
        timeoutMs: PROMPT_ENHANCER_TIMEOUT_MS,
        runId: `prompt-enhancer-${crypto.randomUUID()}`,
        lane: `prompt-enhancer:${params.sessionKey}`,
        extraSystemPrompt: PROMPT_ENHANCER_SYSTEM_PROMPT,
      });
      const assistantText = extractEnhancerReplyText(result.payloads);
      const parsed = parseEnhancerJson(assistantText);
      if (!parsed) {
        throw new Error("model did not return valid JSON");
      }
      return parsed;
    },
  );

  const draft = toPromptDraft({
    output: modelOutput,
    originalPrompt,
    provider: params.provider,
    model: params.model,
  });
  if (!draft) {
    throw new Error("model returned an empty enhancedPrompt");
  }
  return draft;
}

function applyConfirmedPrompt(params: {
  ctx: MsgContext;
  sessionCtx: TemplateContext;
  draft: PromptEnhancerDraft;
}): void {
  params.ctx.Body = params.draft.enhancedPrompt;
  params.ctx.BodyForAgent = params.draft.enhancedPrompt;
  params.sessionCtx.Body = params.draft.enhancedPrompt;
  params.sessionCtx.BodyForAgent = params.draft.enhancedPrompt;
  params.sessionCtx.BodyStripped = params.draft.enhancedPrompt;
}

export async function maybeHandlePromptEnhancer(
  params: PromptEnhancerParams,
): Promise<PromptEnhancerResult | null> {
  const pendingDraft = await resolveActiveDraft({
    sessionEntry: params.sessionEntry,
    sessionStore: params.sessionStore,
    sessionKey: params.sessionKey,
    storePath: params.storePath,
  });
  const promptCommand = parsePromptCommand(params.command.commandBodyNormalized);
  const latestPromptText = resolveLatestPromptText(params.sessionCtx, params.cleanedBody);

  if (pendingDraft) {
    if (promptCommand?.action === "show") {
      return { kind: "reply", reply: buildPromptDraftReply(pendingDraft, "updated") };
    }
    if (promptCommand?.action === "cancel") {
      await persistPromptDraft({
        sessionEntry: params.sessionEntry,
        sessionStore: params.sessionStore,
        sessionKey: params.sessionKey,
        storePath: params.storePath,
        nextDraft: undefined,
      });
      return {
        kind: "reply",
        reply: { text: "Prompt draft canceled. Send a new message when you're ready." },
      };
    }
    if (promptCommand?.action === "run") {
      await persistPromptDraft({
        sessionEntry: params.sessionEntry,
        sessionStore: params.sessionStore,
        sessionKey: params.sessionKey,
        storePath: params.storePath,
        nextDraft: undefined,
      });
      applyConfirmedPrompt({
        ctx: params.ctx,
        sessionCtx: params.sessionCtx,
        draft: pendingDraft,
      });
      return { kind: "continue" };
    }

    const editFeedback =
      promptCommand?.action === "edit" ? (promptCommand.args?.trim() ?? "") : latestPromptText;
    if (promptCommand && promptCommand.action === "edit" && !editFeedback) {
      return {
        kind: "reply",
        reply: {
          text: "Usage: /prompt edit <changes>",
        },
      };
    }
    if (
      promptCommand?.action === "edit" ||
      (!promptCommand && editFeedback && !latestPromptText.startsWith("/"))
    ) {
      try {
        const nextDraft = await runPromptEnhancerModel({
          mode: "edit",
          promptText: editFeedback,
          draft: pendingDraft,
          provider: params.provider,
          model: params.model,
          agentId: params.agentId,
          agentDir: params.agentDir,
          workspaceDir: params.workspaceDir,
          cfg: params.cfg,
          sessionEntry: params.sessionEntry,
          sessionKey: params.sessionKey,
        });
        await persistPromptDraft({
          sessionEntry: params.sessionEntry,
          sessionStore: params.sessionStore,
          sessionKey: params.sessionKey,
          storePath: params.storePath,
          nextDraft,
        });
        return { kind: "reply", reply: buildPromptDraftReply(nextDraft, "updated") };
      } catch (err) {
        return {
          kind: "reply",
          reply: buildPromptEnhancerErrorReply(String(err)),
        };
      }
    }

    return {
      kind: "reply",
      reply: buildPendingDraftReminderReply(),
    };
  }

  if (promptCommand) {
    return { kind: "reply", reply: buildPromptDraftMissingReply() };
  }

  if (!isPromptEnhancerEnabled(params.cfg) || params.opts?.isHeartbeat) {
    return null;
  }
  if (!latestPromptText || hasInboundMedia(params.sessionCtx)) {
    return null;
  }
  if (latestPromptText.startsWith("/")) {
    return null;
  }
  if (isCliProvider(params.provider, params.cfg)) {
    return {
      kind: "reply",
      reply: {
        text: "Prompt enhancement is not available for CLI-backed providers yet.",
        isError: true,
      },
    };
  }

  try {
    const draft = await runPromptEnhancerModel({
      mode: "create",
      promptText: latestPromptText,
      provider: params.provider,
      model: params.model,
      agentId: params.agentId,
      agentDir: params.agentDir,
      workspaceDir: params.workspaceDir,
      cfg: params.cfg,
      sessionEntry: params.sessionEntry,
      sessionKey: params.sessionKey,
    });
    await persistPromptDraft({
      sessionEntry: params.sessionEntry,
      sessionStore: params.sessionStore,
      sessionKey: params.sessionKey,
      storePath: params.storePath,
      nextDraft: draft,
    });
    return { kind: "reply", reply: buildPromptDraftReply(draft, "new") };
  } catch (err) {
    return {
      kind: "reply",
      reply: buildPromptEnhancerErrorReply(String(err)),
    };
  }
}
