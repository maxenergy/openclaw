import { beforeEach, describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../../config/config.js";
import type { PromptEnhancerDraft, SessionEntry } from "../../config/sessions.js";
import type { MsgContext, TemplateContext } from "../templating.js";

const mocks = vi.hoisted(() => ({
  runEmbeddedPiAgent: vi.fn(),
  updateSessionStore: vi.fn(async (..._args: unknown[]) => undefined),
  isCliProvider: vi.fn(() => false),
}));

vi.mock("../../agents/pi-embedded.js", () => ({
  runEmbeddedPiAgent: (...args: unknown[]) => mocks.runEmbeddedPiAgent(...args),
}));

vi.mock("../../agents/model-selection.js", () => ({
  isCliProvider: (...args: unknown[]) => mocks.isCliProvider(...args),
}));

vi.mock("../../config/sessions.js", async () => {
  const actual = await vi.importActual<typeof import("../../config/sessions.js")>(
    "../../config/sessions.js",
  );
  return {
    ...actual,
    updateSessionStore: (...args: unknown[]) => mocks.updateSessionStore(...args),
  };
});

const { maybeHandlePromptEnhancer } = await import("./prompt-enhancer.js");

function buildCfg(enabled = true): OpenClawConfig {
  return {
    agents: {
      defaults: {
        promptEnhancer: {
          enabled,
        },
      },
    },
  } as OpenClawConfig;
}

function buildCommand(overrides: Partial<{ raw: string; normalized: string }> = {}) {
  const raw = overrides.raw ?? "Ship a cleaner prompt";
  const normalized = overrides.normalized ?? raw;
  return {
    surface: "whatsapp",
    channel: "whatsapp",
    ownerList: [],
    senderIsOwner: true,
    isAuthorizedSender: true,
    rawBodyNormalized: raw,
    commandBodyNormalized: normalized,
  };
}

function buildCtx(overrides: Partial<MsgContext> = {}): MsgContext {
  return {
    Body: "Ship a cleaner prompt",
    BodyForAgent: "Ship a cleaner prompt",
    RawBody: "Ship a cleaner prompt",
    CommandBody: "Ship a cleaner prompt",
    From: "whatsapp:+1000",
    To: "whatsapp:+2000",
    Provider: "whatsapp",
    Surface: "whatsapp",
    ChatType: "direct",
    ...overrides,
  };
}

function buildSessionCtx(overrides: Partial<TemplateContext> = {}): TemplateContext {
  return {
    Body: "Ship a cleaner prompt",
    BodyForAgent: "Ship a cleaner prompt",
    BodyStripped: "Ship a cleaner prompt",
    Provider: "whatsapp",
    Surface: "whatsapp",
    ChatType: "direct",
    ...overrides,
  } as TemplateContext;
}

function buildDraft(overrides: Partial<PromptEnhancerDraft> = {}): PromptEnhancerDraft {
  return {
    id: "draft-1",
    createdAt: Date.now(),
    originalPrompt: "Ship a cleaner prompt",
    enhancedPrompt: "Rewrite the onboarding copy and keep all channel docs in sync.",
    goal: "Improve onboarding copy",
    constraints: ["Do not change unrelated docs"],
    assumptions: ["The request only targets docs"],
    clarifyingQuestions: ["Should Telegram examples change too?"],
    modelProvider: "openai",
    model: "gpt-4.1-mini",
    ...overrides,
  };
}

function buildSessionEntry(overrides: Partial<SessionEntry> = {}): SessionEntry {
  return {
    sessionId: "session-1",
    updatedAt: Date.now(),
    sessionFile: "/tmp/does-not-exist.jsonl",
    ...overrides,
  };
}

describe("maybeHandlePromptEnhancer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.isCliProvider.mockReturnValue(false);
  });

  it("creates and stores a prompt draft before execution", async () => {
    mocks.runEmbeddedPiAgent.mockResolvedValue({
      payloads: [
        {
          text: JSON.stringify({
            goal: "Improve onboarding copy",
            constraints: ["Keep existing structure"],
            assumptions: ["The request targets docs only"],
            clarifyingQuestions: [],
            enhancedPrompt:
              "Update the onboarding copy while preserving the current doc structure.",
          }),
        },
      ],
      meta: { durationMs: 12 },
    });

    const sessionEntry = buildSessionEntry();
    const sessionStore: Record<string, SessionEntry> = {
      "agent:main:whatsapp:+1000": sessionEntry,
    };
    const result = await maybeHandlePromptEnhancer({
      ctx: buildCtx(),
      sessionCtx: buildSessionCtx(),
      cfg: buildCfg(true),
      agentId: "main",
      agentDir: "/tmp/agent",
      workspaceDir: "/tmp/workspace",
      sessionEntry,
      sessionStore,
      sessionKey: "agent:main:whatsapp:+1000",
      storePath: "/tmp/sessions.json",
      command: buildCommand(),
      cleanedBody: "Ship a cleaner prompt",
      provider: "openai",
      model: "gpt-4.1-mini",
    });

    expect(result).toEqual(
      expect.objectContaining({
        kind: "reply",
        reply: expect.objectContaining({
          text: expect.stringContaining("Prompt draft ready."),
        }),
      }),
    );
    expect(sessionEntry.promptEnhancerDraft?.enhancedPrompt).toBe(
      "Update the onboarding copy while preserving the current doc structure.",
    );
    expect(mocks.runEmbeddedPiAgent).toHaveBeenCalledWith(
      expect.objectContaining({
        disableTools: true,
        provider: "openai",
        model: "gpt-4.1-mini",
      }),
    );
  });

  it("applies the stored draft when the user confirms with /prompt run", async () => {
    const draft = buildDraft();
    const sessionEntry = buildSessionEntry({ promptEnhancerDraft: draft });
    const sessionStore: Record<string, SessionEntry> = {
      "agent:main:whatsapp:+1000": sessionEntry,
    };
    const ctx = buildCtx({
      Body: "/prompt run",
      BodyForAgent: "/prompt run",
      RawBody: "/prompt run",
      CommandBody: "/prompt run",
    });
    const sessionCtx = buildSessionCtx({
      Body: "/prompt run",
      BodyForAgent: "/prompt run",
      BodyStripped: "/prompt run",
    });

    const result = await maybeHandlePromptEnhancer({
      ctx,
      sessionCtx,
      cfg: buildCfg(true),
      agentId: "main",
      agentDir: "/tmp/agent",
      workspaceDir: "/tmp/workspace",
      sessionEntry,
      sessionStore,
      sessionKey: "agent:main:whatsapp:+1000",
      storePath: "/tmp/sessions.json",
      command: buildCommand({ raw: "/prompt run", normalized: "/prompt run" }),
      cleanedBody: "/prompt run",
      provider: "openai",
      model: "gpt-4.1-mini",
    });

    expect(result).toEqual({ kind: "continue" });
    expect(sessionCtx.BodyStripped).toBe(draft.enhancedPrompt);
    expect(ctx.Body).toBe(draft.enhancedPrompt);
    expect(sessionEntry.promptEnhancerDraft).toBeUndefined();
  });

  it("treats a follow-up plain-text message as prompt draft feedback", async () => {
    mocks.runEmbeddedPiAgent.mockResolvedValue({
      payloads: [
        {
          text: JSON.stringify({
            goal: "Improve onboarding copy",
            constraints: ["Only touch Telegram-related docs"],
            assumptions: [],
            clarifyingQuestions: [],
            enhancedPrompt:
              "Update only the Telegram onboarding docs and keep all other channels unchanged.",
          }),
        },
      ],
      meta: { durationMs: 9 },
    });

    const sessionEntry = buildSessionEntry({ promptEnhancerDraft: buildDraft() });
    const sessionStore: Record<string, SessionEntry> = {
      "agent:main:whatsapp:+1000": sessionEntry,
    };
    const result = await maybeHandlePromptEnhancer({
      ctx: buildCtx({
        Body: "Only touch Telegram docs.",
        BodyForAgent: "Only touch Telegram docs.",
        RawBody: "Only touch Telegram docs.",
        CommandBody: "Only touch Telegram docs.",
      }),
      sessionCtx: buildSessionCtx({
        Body: "Only touch Telegram docs.",
        BodyForAgent: "Only touch Telegram docs.",
        BodyStripped: "Only touch Telegram docs.",
      }),
      cfg: buildCfg(true),
      agentId: "main",
      agentDir: "/tmp/agent",
      workspaceDir: "/tmp/workspace",
      sessionEntry,
      sessionStore,
      sessionKey: "agent:main:whatsapp:+1000",
      storePath: "/tmp/sessions.json",
      command: buildCommand({
        raw: "Only touch Telegram docs.",
        normalized: "Only touch Telegram docs.",
      }),
      cleanedBody: "Only touch Telegram docs.",
      provider: "openai",
      model: "gpt-4.1-mini",
    });

    expect(result).toEqual(
      expect.objectContaining({
        kind: "reply",
        reply: expect.objectContaining({
          text: expect.stringContaining("Prompt draft updated."),
        }),
      }),
    );
    expect(sessionEntry.promptEnhancerDraft?.enhancedPrompt).toBe(
      "Update only the Telegram onboarding docs and keep all other channels unchanged.",
    );
  });

  it("returns an error reply when the enhancer model output is not valid JSON", async () => {
    mocks.runEmbeddedPiAgent.mockResolvedValue({
      payloads: [{ text: "not valid json" }],
      meta: { durationMs: 4 },
    });

    const result = await maybeHandlePromptEnhancer({
      ctx: buildCtx(),
      sessionCtx: buildSessionCtx(),
      cfg: buildCfg(true),
      agentId: "main",
      agentDir: "/tmp/agent",
      workspaceDir: "/tmp/workspace",
      sessionEntry: buildSessionEntry(),
      sessionStore: {
        "agent:main:whatsapp:+1000": buildSessionEntry(),
      },
      sessionKey: "agent:main:whatsapp:+1000",
      storePath: "/tmp/sessions.json",
      command: buildCommand(),
      cleanedBody: "Ship a cleaner prompt",
      provider: "openai",
      model: "gpt-4.1-mini",
    });

    expect(result).toEqual(
      expect.objectContaining({
        kind: "reply",
        reply: expect.objectContaining({
          isError: true,
          text: expect.stringContaining("Prompt enhancement failed"),
        }),
      }),
    );
  });
});
