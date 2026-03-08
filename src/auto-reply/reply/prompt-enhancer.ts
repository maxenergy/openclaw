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
  "Do not wrap the JSON in markdown fences.",
  "Do not add prose before or after the JSON.",
  "Return strict JSON only with this exact schema:",
  '{"goal":"string","constraints":["string"],"assumptions":["string"],"clarifyingQuestions":["string"],"enhancedPrompt":"string"}',
].join("\n");

const PROMPT_ENHANCER_JSON_REPAIR_SYSTEM_PROMPT = [
  "You repair OpenClaw prompt enhancer outputs.",
  "Convert the provided content into strict JSON only.",
  "Preserve the original intent and wording as much as possible.",
  "Do not add markdown fences or any prose outside the JSON object.",
  "Use this exact schema:",
  '{"goal":"string","constraints":["string"],"assumptions":["string"],"clarifyingQuestions":["string"],"enhancedPrompt":"string"}',
  "Use empty arrays when a list field is missing.",
].join("\n");

type PromptEnhancerModelOutput = {
  goal?: string;
  constraints?: string[];
  assumptions?: string[];
  clarifyingQuestions?: string[];
  enhancedPrompt?: string;
};

export type PromptEnhancerResult = { kind: "reply"; reply: ReplyPayload } | { kind: "continue" };
export type PromptEnhancerMode = "off" | "auto" | "manual";

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

export function resolvePromptEnhancerMode(params: {
  cfg: OpenClawConfig;
  sessionEntry?: SessionEntry;
  sessionStore?: Record<string, SessionEntry>;
  sessionKey?: string;
}): PromptEnhancerMode {
  const sessionMode =
    params.sessionEntry?.promptEnhancerMode ??
    (params.sessionKey ? params.sessionStore?.[params.sessionKey]?.promptEnhancerMode : undefined);
  if (sessionMode === "off" || sessionMode === "auto" || sessionMode === "manual") {
    return sessionMode;
  }
  const configuredMode = params.cfg.agents?.defaults?.promptEnhancer?.mode;
  if (configuredMode === "off" || configuredMode === "auto" || configuredMode === "manual") {
    return configuredMode;
  }
  return params.cfg.agents?.defaults?.promptEnhancer?.enabled === true ? "auto" : "off";
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
):
  | { action: "show" | "run" | "cancel" | "edit" | "draft"; args?: string }
  | { action: "mode"; mode: PromptEnhancerMode }
  | null {
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
  if (action === "draft" || action === "create") {
    const nextArgs = args.join(" ").trim();
    return { action: "draft", args: nextArgs || undefined };
  }
  if (action === "off" || action === "auto" || action === "manual") {
    return { action: "mode", mode: action };
  }
  return { action: "draft", args: rest };
}

function replaceSmartQuotes(raw: string): string {
  return raw
    .replaceAll("\u201c", '"')
    .replaceAll("\u201d", '"')
    .replaceAll("\u2018", "'")
    .replaceAll("\u2019", "'");
}

function escapeJsonString(raw: string): string {
  return JSON.stringify(raw.replaceAll("\\'", "'"));
}

function collectBalancedJsonObjects(raw: string): string[] {
  const results: string[] = [];
  let depth = 0;
  let start = -1;
  let stringQuote: '"' | "'" | null = null;
  let escaped = false;

  for (let index = 0; index < raw.length; index += 1) {
    const char = raw[index];
    if (stringQuote) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === stringQuote) {
        stringQuote = null;
      }
      continue;
    }
    if (char === '"' || char === "'") {
      stringQuote = char;
      continue;
    }
    if (char === "{") {
      if (depth === 0) {
        start = index;
      }
      depth += 1;
      continue;
    }
    if (char === "}") {
      if (depth === 0) {
        continue;
      }
      depth -= 1;
      if (depth === 0 && start !== -1) {
        results.push(raw.slice(start, index + 1).trim());
        start = -1;
      }
    }
  }
  return results;
}

function normalizeJsonishCandidate(raw: string): string[] {
  const trimmed = replaceSmartQuotes(raw.trim().replace(/^\uFEFF/, ""));
  if (!trimmed) {
    return [];
  }

  const variants = new Set<string>([trimmed]);
  const withoutTrailingCommas = trimmed.replace(/,\s*([}\]])/g, "$1");
  variants.add(withoutTrailingCommas);

  const bareKeysQuoted = withoutTrailingCommas.replace(
    /([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)/g,
    '$1"$2"$3',
  );
  variants.add(bareKeysQuoted);

  const singleQuotedKeys = bareKeysQuoted.replace(
    /'([^'\\]*(?:\\.[^'\\]*)*)'(?=\s*:)/g,
    (_match, inner: string) => escapeJsonString(inner),
  );
  variants.add(singleQuotedKeys);

  const singleQuotedValues = singleQuotedKeys
    .replace(
      /(:\s*)'([^'\\]*(?:\\.[^'\\]*)*)'(?=\s*[,}\]])/g,
      (_match, prefix: string, inner: string) => `${prefix}${escapeJsonString(inner)}`,
    )
    .replace(
      /([[,]\s*)'([^'\\]*(?:\\.[^'\\]*)*)'(?=\s*[,}\]])/g,
      (_match, prefix: string, inner: string) => `${prefix}${escapeJsonString(inner)}`,
    );
  variants.add(singleQuotedValues);

  return [...variants];
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
  for (const candidate of candidates) {
    for (const extractedObject of collectBalancedJsonObjects(candidate)) {
      candidates.add(extractedObject);
    }
  }

  for (const candidate of candidates) {
    for (const variant of normalizeJsonishCandidate(candidate)) {
      try {
        const parsed = JSON.parse(variant) as unknown;
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          continue;
        }
        return parsed as PromptEnhancerModelOutput;
      } catch {
        continue;
      }
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

function buildRepairPrompt(originalPrompt: string, invalidOutput: string): string {
  return [
    "The previous prompt-enhancer model output was not valid strict JSON.",
    "Recover it into the required schema without changing the user's intent.",
    "",
    "Original user request:",
    originalPrompt.trim(),
    "",
    "Invalid model output:",
    invalidOutput.trim(),
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

function buildPromptEnhancerStatusReply(mode: PromptEnhancerMode): ReplyPayload {
  const modeText =
    mode === "auto"
      ? "Prompt enhancer mode: auto. Normal text messages will be drafted before execution."
      : mode === "manual"
        ? "Prompt enhancer mode: manual. Normal text messages execute directly unless you explicitly create a draft."
        : "Prompt enhancer mode: off. Messages execute directly without prompt drafting.";
  return {
    text: [
      modeText,
      "Switch with `/prompt auto`, `/prompt manual`, or `/prompt off`.",
      "Create a draft with `/prompt <request>` or `/prompt draft <request>`.",
      "Review a pending draft with `/prompt show`, then `/prompt run|edit|cancel`.",
    ].join("\n\n"),
  };
}

function buildPromptDraftMissingReply(mode: PromptEnhancerMode): ReplyPayload {
  return {
    text:
      mode === "manual"
        ? "No pending prompt draft. Use `/prompt <request>` or `/prompt draft <request>` to create one."
        : "No pending prompt draft. Send a normal message first to generate one.",
  };
}

function buildPendingDraftReminderReply(): ReplyPayload {
  return {
    text: "A prompt draft is waiting for review. Reply with `/prompt run`, `/prompt edit <changes>`, `/prompt show`, or `/prompt cancel`.",
  };
}

function buildPromptEnhancerDisabledReply(): ReplyPayload {
  return {
    text: "Prompt enhancement is off for this session. Use `/prompt manual` or `/prompt auto`, or set `agents.defaults.promptEnhancer.mode` to `manual` or `auto`.",
    isError: true,
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

  let assistantText = "";
  let modelOutput = await withEnhancerSessionFile(
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
      assistantText = extractEnhancerReplyText(result.payloads);
      return parseEnhancerJson(assistantText);
    },
  );
  let draft = modelOutput
    ? toPromptDraft({
        output: modelOutput,
        originalPrompt,
        provider: params.provider,
        model: params.model,
      })
    : null;

  if (!modelOutput || !draft) {
    modelOutput = await withEnhancerSessionFile(
      params.sessionEntry?.sessionFile,
      async (sessionFile, sessionId) => {
        const result = await runEmbeddedPiAgent({
          sessionId,
          sessionFile,
          agentId: params.agentId,
          workspaceDir: params.workspaceDir,
          agentDir: params.agentDir,
          config: params.cfg,
          prompt: buildRepairPrompt(originalPrompt, assistantText || params.promptText),
          provider: params.provider,
          model: params.model,
          disableTools: true,
          thinkLevel: "off",
          reasoningLevel: "off",
          verboseLevel: "off",
          timeoutMs: PROMPT_ENHANCER_TIMEOUT_MS,
          runId: `prompt-enhancer-repair-${crypto.randomUUID()}`,
          lane: `prompt-enhancer-repair:${params.sessionKey}`,
          extraSystemPrompt: PROMPT_ENHANCER_JSON_REPAIR_SYSTEM_PROMPT,
        });
        return parseEnhancerJson(extractEnhancerReplyText(result.payloads));
      },
    );
    draft = modelOutput
      ? toPromptDraft({
          output: modelOutput,
          originalPrompt,
          provider: params.provider,
          model: params.model,
        })
      : null;
  }

  if (!draft) {
    throw new Error("model did not return usable JSON");
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
  params.ctx.RawBody = params.draft.enhancedPrompt;
  params.ctx.CommandBody = params.draft.enhancedPrompt;
  params.ctx.BodyForCommands = params.draft.enhancedPrompt;
  params.sessionCtx.Body = params.draft.enhancedPrompt;
  params.sessionCtx.BodyForAgent = params.draft.enhancedPrompt;
  params.sessionCtx.RawBody = params.draft.enhancedPrompt;
  params.sessionCtx.CommandBody = params.draft.enhancedPrompt;
  params.sessionCtx.BodyForCommands = params.draft.enhancedPrompt;
  params.sessionCtx.BodyStripped = params.draft.enhancedPrompt;
}

export async function maybeHandlePromptEnhancer(
  params: PromptEnhancerParams,
): Promise<PromptEnhancerResult | null> {
  const mode = resolvePromptEnhancerMode({
    cfg: params.cfg,
    sessionEntry: params.sessionEntry,
    sessionStore: params.sessionStore,
    sessionKey: params.sessionKey,
  });
  const pendingDraft = await resolveActiveDraft({
    sessionEntry: params.sessionEntry,
    sessionStore: params.sessionStore,
    sessionKey: params.sessionKey,
    storePath: params.storePath,
  });
  const promptCommand = parsePromptCommand(params.command.commandBodyNormalized);
  const latestPromptText = resolveLatestPromptText(params.sessionCtx, params.cleanedBody);

  if (pendingDraft) {
    if (promptCommand?.action === "mode") {
      return { kind: "reply", reply: buildPromptEnhancerStatusReply(mode) };
    }
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

    if (mode === "off") {
      if (promptCommand?.action === "edit" || promptCommand?.action === "draft") {
        return { kind: "reply", reply: buildPromptEnhancerDisabledReply() };
      }
      return null;
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
    if (promptCommand.action === "mode") {
      return { kind: "reply", reply: buildPromptEnhancerStatusReply(mode) };
    }
    if (promptCommand.action === "show") {
      return { kind: "reply", reply: buildPromptEnhancerStatusReply(mode) };
    }
    if (promptCommand.action === "draft") {
      if (mode === "off") {
        return { kind: "reply", reply: buildPromptEnhancerDisabledReply() };
      }
      const promptText = promptCommand.args?.trim() ?? "";
      if (!promptText) {
        return {
          kind: "reply",
          reply: {
            text: "Usage: /prompt <request> or /prompt draft <request>",
          },
        };
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
          promptText,
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
    return {
      kind: "reply",
      reply: buildPromptDraftMissingReply(mode),
    };
  }

  if (mode !== "auto" || params.opts?.isHeartbeat) {
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
