import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SessionEntry } from "../../config/sessions.js";
import type { TemplateContext } from "../templating.js";
import { clearInlineDirectives } from "./get-reply-directives-utils.js";
import { buildTestCtx } from "./test-ctx.js";
import type { TypingController } from "./typing.js";

const mocks = vi.hoisted(() => ({
  handleCommands: vi.fn(),
  runEmbeddedPiAgent: vi.fn(),
  updateSessionStore: vi.fn(async (..._args: unknown[]) => undefined),
  isCliProvider: vi.fn(() => false),
}));

vi.mock("./commands.js", () => ({
  handleCommands: (...args: unknown[]) => mocks.handleCommands(...args),
  buildStatusReply: vi.fn(),
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

const { handleInlineActions } = await import("./get-reply-inline-actions.js");
type HandleInlineActionsInput = Parameters<typeof handleInlineActions>[0];

const createTypingController = (): TypingController => ({
  onReplyStart: async () => {},
  startTypingLoop: async () => {},
  startTypingOnText: async () => {},
  refreshTypingTtl: () => {},
  isActive: () => false,
  markRunComplete: () => {},
  markDispatchIdle: () => {},
  cleanup: vi.fn(),
});

const createHandleInlineActionsInput = (params: {
  ctx: ReturnType<typeof buildTestCtx>;
  typing: TypingController;
  cleanedBody: string;
  command?: Partial<HandleInlineActionsInput["command"]>;
  overrides?: Partial<Omit<HandleInlineActionsInput, "ctx" | "sessionCtx" | "typing" | "command">>;
}): HandleInlineActionsInput => {
  const baseCommand: HandleInlineActionsInput["command"] = {
    surface: "whatsapp",
    channel: "whatsapp",
    channelId: "whatsapp",
    ownerList: [],
    senderIsOwner: true,
    isAuthorizedSender: true,
    senderId: "sender-1",
    abortKey: "whatsapp:+999",
    rawBodyNormalized: params.cleanedBody,
    commandBodyNormalized: params.cleanedBody,
    from: "whatsapp:+999",
    to: "whatsapp:+999",
  };
  return {
    ctx: params.ctx,
    sessionCtx: params.ctx as unknown as TemplateContext,
    cfg: {},
    agentId: "main",
    sessionKey: "s:main",
    workspaceDir: "/tmp",
    isGroup: false,
    typing: params.typing,
    allowTextCommands: false,
    inlineStatusRequested: false,
    command: {
      ...baseCommand,
      ...params.command,
    },
    directives: clearInlineDirectives(params.cleanedBody),
    cleanedBody: params.cleanedBody,
    elevatedEnabled: false,
    elevatedAllowed: false,
    elevatedFailures: [],
    defaultActivation: () => "always",
    resolvedThinkLevel: undefined,
    resolvedVerboseLevel: undefined,
    resolvedReasoningLevel: "off",
    resolvedElevatedLevel: "off",
    resolveDefaultThinkingLevel: async () => "off",
    provider: "openai",
    model: "gpt-4.1-mini",
    contextTokens: 0,
    abortedLastRun: false,
    sessionScope: "per-sender",
    ...params.overrides,
  };
};

describe("handleInlineActions prompt enhancer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.handleCommands.mockResolvedValue({ shouldContinue: true });
    mocks.isCliProvider.mockReturnValue(false);
  });

  it("creates a prompt draft from the stripped /new tail before execution", async () => {
    mocks.runEmbeddedPiAgent.mockResolvedValue({
      payloads: [
        {
          text: JSON.stringify({
            goal: "Deploy the service",
            constraints: ["Keep the existing rollout process"],
            assumptions: [],
            clarifyingQuestions: [],
            enhancedPrompt:
              "Deploy the service, keep the existing rollout process, and verify the deployment result.",
          }),
        },
      ],
      meta: { durationMs: 12 },
    });

    const typing = createTypingController();
    const sessionEntry: SessionEntry = {
      sessionId: "session-1",
      updatedAt: Date.now(),
      sessionFile: "/tmp/does-not-exist.jsonl",
    };
    const sessionStore = { "s:main": sessionEntry };
    const ctx = buildTestCtx({
      Body: "/new continue with deployment",
      BodyForAgent: "/new continue with deployment",
      BodyForCommands: "/new continue with deployment",
      RawBody: "/new continue with deployment",
      CommandBody: "/new continue with deployment",
    });

    const result = await handleInlineActions(
      createHandleInlineActionsInput({
        ctx,
        typing,
        cleanedBody: "continue with deployment",
        command: {
          rawBodyNormalized: "/new continue with deployment",
          commandBodyNormalized: "/new continue with deployment",
        },
        overrides: {
          cfg: {
            agents: {
              defaults: {
                promptEnhancer: {
                  enabled: true,
                },
              },
            },
          },
          sessionEntry,
          sessionStore,
          storePath: "/tmp/sessions.json",
        },
      }),
    );

    expect(result).toEqual(
      expect.objectContaining({
        kind: "reply",
        reply: expect.objectContaining({
          text: expect.stringContaining("Prompt draft ready."),
        }),
      }),
    );
    expect(mocks.handleCommands).toHaveBeenCalledTimes(1);
    expect(mocks.runEmbeddedPiAgent).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: expect.stringContaining("continue with deployment"),
      }),
    );
    expect(sessionEntry.promptEnhancerDraft?.enhancedPrompt).toBe(
      "Deploy the service, keep the existing rollout process, and verify the deployment result.",
    );
  });

  it("does not intercept explicit skill commands with a prompt draft", async () => {
    const typing = createTypingController();
    const ctx = buildTestCtx({
      Body: "/skill demo-skill tighten the onboarding docs",
      BodyForAgent: "/skill demo-skill tighten the onboarding docs",
      BodyForCommands: "/skill demo-skill tighten the onboarding docs",
      RawBody: "/skill demo-skill tighten the onboarding docs",
      CommandBody: "/skill demo-skill tighten the onboarding docs",
    });

    const result = await handleInlineActions(
      createHandleInlineActionsInput({
        ctx,
        typing,
        cleanedBody: "/skill demo-skill tighten the onboarding docs",
        command: {
          rawBodyNormalized: "/skill demo-skill tighten the onboarding docs",
          commandBodyNormalized: "/skill demo-skill tighten the onboarding docs",
        },
        overrides: {
          allowTextCommands: true,
          cfg: {
            agents: {
              defaults: {
                promptEnhancer: {
                  enabled: true,
                },
              },
            },
          },
          skillCommands: [
            {
              name: "demo-skill",
              skillName: "demo-skill",
              description: "Demo skill",
            },
          ],
        },
      }),
    );

    expect(result).toEqual(
      expect.objectContaining({
        kind: "continue",
      }),
    );
    expect(mocks.runEmbeddedPiAgent).not.toHaveBeenCalled();
    expect(ctx.Body).toContain('Use the "demo-skill" skill');
  });
});
