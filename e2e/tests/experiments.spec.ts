import { test, expect } from "@playwright/test";

test.describe("Experiments Page", () => {
  test("should load experiments page with filters and data", async ({ page }) => {
    await page.goto("/experiments");

    // Check page loaded
    await expect(
      page.getByRole("heading", { name: "Experiments", exact: true })
    ).toBeVisible();
    await expect(
      page.getByText("Browse and filter all Gren experiments")
    ).toBeVisible();

    // Check filter inputs exist
    await expect(page.getByPlaceholder("Filter by namespace...")).toBeVisible();
    await expect(page.getByRole("combobox").first()).toBeVisible();

    // Should have 11 experiments from generate_data.py
    await expect(page.getByText(/Showing \d+ of 11 experiments/)).toBeVisible();

    // Check that experiment cards display real class names
    const experimentClasses = ["PrepareDataset", "TrainModel", "TrainTextModel"];
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

  test("should filter by result status", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for initial data to load
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();

    // Select success filter
    const resultStatusSelect = page.getByRole("combobox").first();
    await resultStatusSelect.selectOption("success");

    // Should show 6 successful experiments
    await expect(page.getByText(/Showing \d+ of 6 experiments/)).toBeVisible();
  });

  test("should filter by namespace", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for initial load
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();

    // Enter namespace filter
    const namespaceInput = page.getByPlaceholder("Filter by namespace...");
    await namespaceInput.fill("__main__.TrainModel");

    // Wait for filter to apply
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();
  });

  test("should show migration tag in list", async ({ page }) => {
    await page.goto("/experiments");
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();

    await expect(page.locator("span", { hasText: "alias" })).toBeVisible();
  });

  test("should show migration toggle and link on detail", async ({ page }) => {
    await page.goto("/experiments");
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();

    const migrationRow = page.locator("tr", { has: page.getByText("alias") }).first();
    await migrationRow.locator("a").first().click();

    await expect(page.getByText("View original")).toBeVisible();
    await expect(page.getByText("Original")).toBeVisible();
    await expect(page.getByText("Aliased")).toBeVisible();

    await page.getByText("Original").click();
    await expect(page.getByText("Original status:")).toBeVisible();
  });

  test("should handle empty results gracefully", async ({ page }) => {
    await page.goto("/experiments");

    // Wait for initial data to load
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();

    // Filter with unlikely namespace
    const namespaceInput = page.getByPlaceholder("Filter by namespace...");
    await namespaceInput.fill("nonexistent_namespace_xyz");

    // Should show empty state message
    await expect(page.getByText("No experiments found")).toBeVisible();
  });
});
