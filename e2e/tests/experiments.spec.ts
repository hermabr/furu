import { test, expect } from "@playwright/test";

test.describe("Experiments Page", () => {
  test("should load experiments page with filters and data", async ({ page }) => {
    await page.goto("/experiments");

    // Check page loaded
    await expect(
      page.getByRole("heading", { name: "Experiments", exact: true })
    ).toBeVisible();
    await expect(
      page.getByText("Browse and filter all Furu experiments")
    ).toBeVisible();

    // Check filter inputs exist
    await expect(page.getByPlaceholder("Filter by namespace...")).toBeVisible();
    await expect(page.getByRole("combobox").first()).toBeVisible();

    // Should have 14 experiments from generate_data.py
    await expect(page.getByText(/Showing \d+ of 14 experiments/)).toBeVisible();

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

    // Should show 8 successful experiments
    await expect(page.getByText(/Showing \d+ of 8 experiments/)).toBeVisible();
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

    const aliasBadges = page.locator("span", { hasText: "alias" });
    await expect(aliasBadges).toHaveCount(2);
  });

  test("should show migration toggle and link on detail", async ({ page }) => {
    await page.goto("/experiments");
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();

    const migrationRow = page.locator("tr", { has: page.getByText("alias") }).first();
    await migrationRow.locator("a").first().click();

    await expect(page.getByRole("link", { name: "View original" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Original" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Aliased" })).toBeVisible();

    await page.getByRole("button", { name: "Original" }).click();
    await expect(page.getByText("Original status:")).toBeVisible();
  });

  test("should show alias links from original", async ({ page }) => {
    await page.goto("/experiments");
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();

    const aliasRow = page.locator("tr", { has: page.getByText("alias") }).first();
    await aliasRow.locator("a").first().click();

    const originalLink = page.getByRole("link", { name: "View original" });
    await expect(originalLink).toBeVisible();
    await originalLink.click();

    const expectedAliasHashes = [
      "91e3f929e8cfee3288cf",
      "c274e94e4acae91dd6b7",
    ].sort();

    await expect
      .poll(async () => {
        const response = await page.request.get(
          "/api/experiments/my_project.pipelines.PrepareDataset/538934772119a51b05c3?view=resolved"
        );
        if (!response.ok()) {
          return null;
        }
        const data = await response.json();
        return data.alias_hashes ? [...data.alias_hashes].sort() : null;
      })
      .toEqual(expectedAliasHashes);

    await page.waitForLoadState("networkidle");

    await expect
      .poll(async () => page.getByRole("link", { name: "View alias" }).count())
      .toBe(2);

    const aliasLinks = page.getByRole("link", { name: "View alias" });

    await aliasLinks.first().click({ force: true });
    await expect(page.getByRole("link", { name: "View original" })).toBeVisible();
  });

  test("moved detail shows original link", async ({ page }) => {
    await page.goto("/experiments");
    await expect(page.getByText(/Showing \d+ of \d+ experiments/)).toBeVisible();

    const movedRow = page.locator("tr", { has: page.getByText("moved") }).first();
    await movedRow.locator("a").first().click();

    const originalLink = page.getByRole("link", { name: "View original" });
    if ((await originalLink.count()) > 0) {
      await expect(originalLink).toBeVisible({ timeout: 5000 });
      await expect(page.getByRole("button", { name: "Original" })).toBeVisible();
      await page.getByRole("button", { name: "Original" }).click({ force: true });
      return;
    }

    const originalButton = page.getByRole("button", { name: "Original" });
    await expect(originalButton).toBeVisible();
    if (await originalButton.isEnabled()) {
      await originalButton.click({ force: true });
      await expect(page.getByRole("link", { name: "View original" })).toBeVisible();
    }
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
