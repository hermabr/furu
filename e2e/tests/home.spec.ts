import { test, expect } from "@playwright/test";

test.describe("Dashboard Home Page", () => {
  test("should load the dashboard", async ({ page }) => {
    await page.goto("/");

    // Check that the page title is visible
    await expect(
      page.getByRole("heading", { name: "Dashboard" })
    ).toBeVisible();
  });

  test("should display API health status", async ({ page }) => {
    await page.goto("/");

    // Wait for the API status to load
    await expect(page.getByText("API Status:")).toBeVisible();
    await expect(page.getByText("Healthy")).toBeVisible();
  });

  test("should display stats cards with generated data", async ({ page }) => {
    await page.goto("/");

    // Check that stats cards are visible
    await expect(page.getByText("Total Experiments")).toBeVisible();
    await expect(page.getByText("Running")).toBeVisible();
    await expect(page.getByText("Successful")).toBeVisible();
    // Use exact match to avoid matching status badges
    await expect(page.getByText("Failed", { exact: true })).toBeVisible();

    // The stats should reflect our generated data (10 total experiments)
    // We can't easily check specific numbers in the UI without knowing the exact structure
    // but we should see non-zero values for successful experiments
    await expect(page.getByText("10")).toBeVisible({ timeout: 10000 });
  });

  test("should have working navigation", async ({ page }) => {
    await page.goto("/");

    // Check navigation links exist
    await expect(page.getByRole("link", { name: "Dashboard" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Experiments" })).toBeVisible();
  });

  test("should show version number", async ({ page }) => {
    await page.goto("/");

    // Wait for the version to load
    await expect(page.getByText(/v\d+\.\d+\.\d+/)).toBeVisible();
  });

  test("should show recent experiments or activity", async ({ page }) => {
    await page.goto("/");

    // Wait for the page to fully load with data
    await expect(page.getByText("Total Experiments")).toBeVisible();

    // Should show some content related to experiments
    // This could be a recent activity section or experiment count
    await expect(page.getByText("10")).toBeVisible({ timeout: 10000 });
  });
});
