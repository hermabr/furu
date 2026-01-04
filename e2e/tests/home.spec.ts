import { test, expect } from "@playwright/test";

test.describe("Dashboard Home Page", () => {
  test("should load dashboard with stats and health status", async ({ page }) => {
    await page.goto("/");

    // Check page title
    await expect(
      page.getByRole("heading", { name: "Dashboard" })
    ).toBeVisible();

    // Check API health status
    await expect(page.getByText("API Status:")).toBeVisible();
    await expect(page.getByText("Healthy")).toBeVisible();

    // Check version number
    await expect(page.getByText(/v\d+\.\d+\.\d+/)).toBeVisible();

    // Check stats cards are visible with correct data
    await expect(page.getByText("Total Experiments")).toBeVisible();
    await expect(page.getByText("Running")).toBeVisible();
    await expect(page.getByText("Successful")).toBeVisible();
    await expect(page.getByText("Failed", { exact: true })).toBeVisible();

    // Verify stats value matches generated data (10 experiments)
    await expect(page.getByTestId("stats-total-value")).toHaveText("10");
  });

  test("should have working navigation", async ({ page }) => {
    await page.goto("/");

    // Check navigation links exist
    await expect(page.getByRole("link", { name: "Dashboard" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Experiments" })).toBeVisible();
  });
});
