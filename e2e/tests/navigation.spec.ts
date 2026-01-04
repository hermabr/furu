import { test, expect } from "@playwright/test";

test.describe("Navigation", () => {
  test("should navigate between pages", async ({ page }) => {
    // Start at home
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();

    // Navigate to experiments
    await page.getByRole("link", { name: "Experiments" }).click();
    await expect(page).toHaveURL("/experiments");
    await expect(
      page.getByRole("heading", { name: "Experiments", exact: true })
    ).toBeVisible();

    // Navigate back via Dashboard link
    await page.getByRole("link", { name: "Dashboard" }).click();
    await expect(page).toHaveURL("/");

    // Navigate via logo
    await page.getByRole("link", { name: "Experiments" }).click();
    await page.getByRole("link", { name: "Huldra" }).click();
    await expect(page).toHaveURL("/");
  });
});

