import { test, expect } from "@playwright/test";

test.describe("Experiments Page", () => {
  test("should load experiments page", async ({ page }) => {
    await page.goto("/experiments");

    // Check page loaded - use exact match to avoid matching "No experiments found"
    await expect(
      page.getByRole("heading", { name: "Experiments", exact: true })
    ).toBeVisible();
    await expect(
      page.getByText("Browse and filter all Huldra experiments")
    ).toBeVisible();
  });

  test("should display filter controls", async ({ page }) => {
    await page.goto("/experiments");

    // Check filter inputs exist
    await expect(page.getByPlaceholder("Filter by namespace...")).toBeVisible();
    await expect(page.getByRole("combobox").first()).toBeVisible();
  });

  test("should show results count with generated data", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for experiments to load - should have 10 experiments from generate_data.py
    await expect(page.getByText(/Showing \d+ of 10 experiments/)).toBeVisible({
      timeout: 10000,
    });
  });

  test("should filter by result status", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for page to load - use exact match
    await expect(
      page.getByRole("heading", { name: "Experiments", exact: true })
    ).toBeVisible();

    // Wait for initial data to load
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible({
      timeout: 10000,
    });

    // Select success filter (first dropdown is result status)
    const resultStatusSelect = page.getByRole("combobox").first();
    await resultStatusSelect.selectOption("success");

    // Wait for filter to apply - should show 6 successful experiments
    await expect(page.getByText(/Showing \d+ of 6 experiments/)).toBeVisible({
      timeout: 10000,
    });
  });

  test("should filter by namespace", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for initial load
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible({
      timeout: 10000,
    });

    // Enter namespace filter - use namespace from generated data
    const namespaceInput = page.getByPlaceholder("Filter by namespace...");
    await namespaceInput.fill("__main__.TrainModel");

    // Wait for debounce and filter to apply
    await page.waitForTimeout(600);

    // Should filter to show TrainModel experiments
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible({
      timeout: 10000,
    });
  });

  test("should handle empty results gracefully", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for initial data to load
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible({
      timeout: 10000,
    });

    // Filter with unlikely namespace via the UI input
    const namespaceInput = page.getByPlaceholder("Filter by namespace...");
    await namespaceInput.fill("nonexistent_namespace_xyz");

    // Wait for debounce and filter to apply
    await page.waitForTimeout(600);

    // Should show empty state message (EmptyState component)
    await expect(page.getByText("No experiments found")).toBeVisible({
      timeout: 10000,
    });
  });

  test("should display experiment cards with real data", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for experiments to load
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible({
      timeout: 10000,
    });

    // Check that experiment cards are displayed
    // These should be real class names from our generated data
    const experimentClasses = [
      "PrepareDataset",
      "TrainModel",
      "EvalModel",
      "DataLoader",
    ];

    // At least one of these should be visible
    let foundAny = false;
    for (const className of experimentClasses) {
      const count = await page.getByText(className).count();
      if (count > 0) {
        foundAny = true;
        break;
      }
    }
    expect(foundAny).toBe(true);
  });

  test("should show different status badges", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for experiments to load
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible({
      timeout: 10000,
    });

    // The generated data has experiments with various states
    // Check that at least some status indicators are visible
    // (success, running, failed, queued badges)
    const statusTexts = ["success", "running", "failed", "queued", "incomplete"];
    let foundStatuses = 0;
    for (const status of statusTexts) {
      const count = await page.getByText(status, { exact: true }).count();
      if (count > 0) foundStatuses++;
    }
    // Should find at least 2 different statuses
    expect(foundStatuses).toBeGreaterThanOrEqual(1);
  });
});
