import * as fs from "fs";
import * as os from "os";
import * as path from "path";

import { defineConfig, devices } from "@playwright/test";

const e2eDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "furu-e2e-"));
process.env.FURU_PATH = e2eDataDir;
process.env.FURU_E2E_DATA_DIR = e2eDataDir;

/**
 * Playwright configuration for Furu Dashboard E2E tests.
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [["html"], ["list"]] : [["list"]],

  // Global timeout for each test
  timeout: 15000,

  // Expect timeout - reduced since local server is fast
  expect: {
    timeout: 2000,
  },

  use: {
    baseURL: "http://localhost:8000",
    // Only capture traces/screenshots on failure in CI
    trace: process.env.CI ? "on-first-retry" : "off",
    screenshot: process.env.CI ? "only-on-failure" : "off",
    video: process.env.CI ? "on-first-retry" : "off",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  /* Global setup: Generate test data before running tests */
  globalSetup: "./global-setup.ts",

  /* Global teardown: Cleanup temp e2e data */
  globalTeardown: "./global-teardown.ts",

  /* Run the dashboard server before starting the tests */
  webServer: {
    command: "cd .. && uv run python -m furu.dashboard serve --port 8000",
    url: "http://localhost:8000/api/health",
    reuseExistingServer: false,
    timeout: 30000,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      FURU_PATH: e2eDataDir,
      FURU_E2E_DATA_DIR: e2eDataDir,
    },
  },
});
