import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for Huldra Dashboard E2E tests.
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

  /* Run the dashboard server before starting the tests */
  webServer: {
    command: "cd .. && uv run python -m huldra.dashboard serve --port 8000",
    url: "http://localhost:8000/api/health",
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
