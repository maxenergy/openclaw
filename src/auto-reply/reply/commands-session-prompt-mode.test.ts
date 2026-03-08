import { describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../../config/config.js";
import type { SessionEntry } from "../../config/sessions.js";
import { handlePromptModeCommand } from "./commands-session.js";
import { buildCommandTestParams } from "./commands.test-harness.js";

const baseCfg = {
  agents: {
    defaults: {
      promptEnhancer: {
        mode: "auto",
      },
    },
  },
} satisfies OpenClawConfig;

function buildParams(commandBody: string, sessionEntry?: SessionEntry) {
  const params = buildCommandTestParams(commandBody, baseCfg);
  if (sessionEntry) {
    params.sessionEntry = sessionEntry;
    params.sessionStore = {
      [params.sessionKey]: sessionEntry,
    };
  }
  return params;
}

describe("handlePromptModeCommand", () => {
  it("sets the prompt enhancer mode for the current session and clears pending drafts", async () => {
    const sessionEntry: SessionEntry = {
      sessionId: "session-1",
      updatedAt: Date.now(),
      promptEnhancerMode: "manual",
      promptEnhancerDraft: {
        id: "draft-1",
        createdAt: Date.now(),
        originalPrompt: "Ship a cleaner prompt",
        enhancedPrompt: "Rewrite the onboarding docs with clearer steps.",
      },
    };

    const result = await handlePromptModeCommand(buildParams("/prompt off", sessionEntry), true);

    expect(result).toEqual({
      shouldContinue: false,
      reply: {
        text: "⚙️ Prompt enhancer mode set to off for this session. Pending draft canceled.",
      },
    });
    expect(sessionEntry.promptEnhancerMode).toBe("off");
    expect(sessionEntry.promptEnhancerDraft).toBeUndefined();
  });

  it("returns usage when extra args are provided", async () => {
    const result = await handlePromptModeCommand(buildParams("/prompt auto now"), true);

    expect(result).toEqual({
      shouldContinue: false,
      reply: { text: "⚙️ Usage: /prompt auto|manual|off" },
    });
  });

  it("does not intercept non-mode prompt commands", async () => {
    const result = await handlePromptModeCommand(
      buildParams("/prompt draft tighten the README"),
      true,
    );

    expect(result).toBeNull();
  });
});
