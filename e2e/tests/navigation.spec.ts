import { test, expect } from "@playwright/test";

test.describe("Navigation", () => {
  test.describe("Sidebar Navigation", () => {
    test("should navigate between main pages via sidebar", async ({ page }) => {
      // Start at home
      await page.goto("/");
      await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();

      // Navigate to experiments
      await page.getByRole("link", { name: "Experiments" }).click();
      await expect(page).toHaveURL("/experiments");
      await expect(
        page.getByRole("heading", { name: "Experiments", exact: true })
      ).toBeVisible();

      // Navigate to DAG
      await page.getByRole("link", { name: "DAG" }).click();
      await expect(page).toHaveURL("/dag");
      await expect(
        page.getByRole("heading", { name: "Experiment DAG" })
      ).toBeVisible();

      // Navigate back to Dashboard
      await page.getByRole("link", { name: "Dashboard" }).click();
      await expect(page).toHaveURL("/");
      await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
    });

    test("should navigate home via logo", async ({ page }) => {
      await page.goto("/experiments");
      await expect(page).toHaveURL("/experiments");

      // Click logo to go home
      await page.getByRole("link", { name: "Furu" }).click();
      await expect(page).toHaveURL("/");
    });
  });

  test.describe("Experiment List to Detail Navigation", () => {
    test("should navigate to experiment detail from experiments list", async ({
      page,
    }) => {
      await page.goto("/experiments");

      // Wait for experiments to load
      await expect(
        page.getByText(/Showing \d+ of \d+ experiments/)
      ).toBeVisible();

      // Get the first experiment link in the table
      const firstExperimentLink = page
        .locator("table tbody tr")
        .first()
        .locator("a")
        .first();
      const className = await firstExperimentLink.textContent();

      // Click to navigate to detail
      await firstExperimentLink.click();

      // Should navigate to detail page URL pattern
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);

      // Should show detail view content (not the list page)
      await expect(
        page.getByText("Browse and filter all Furu experiments")
      ).not.toBeVisible();

      // Should show the experiment class name in the header
      await expect(
        page.getByRole("heading", { name: className || "" })
      ).toBeVisible();

      // Should show configuration section
      await expect(page.getByRole("heading", { name: "Configuration" })).toBeVisible();
    });

    test("should navigate back to experiments list via breadcrumb", async ({
      page,
    }) => {
      await page.goto("/experiments");

      // Wait for experiments and click first one
      await expect(
        page.getByText(/Showing \d+ of \d+ experiments/)
      ).toBeVisible();
      await page.locator("table tbody tr").first().locator("a").first().click();

      // Verify we're on detail page
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);

      // Click breadcrumb to go back
      await page
        .locator("a", { hasText: "Experiments" })
        .filter({ has: page.locator(":scope:not(:has-text('View all'))") })
        .first()
        .click();

      // Should be back on experiments list
      await expect(page).toHaveURL("/experiments");
      await expect(
        page.getByText("Browse and filter all Furu experiments")
      ).toBeVisible();
    });
  });

  test.describe("Home Page to Detail Navigation", () => {
    test("should navigate to experiment detail from home recent experiments", async ({
      page,
    }) => {
      await page.goto("/");

      // Wait for recent experiments to load
      await expect(
        page.getByRole("heading", { name: "Recent Experiments" })
      ).toBeVisible();

      // Find experiment links in the recent experiments table
      const recentExperimentsTable = page.locator("table").first();
      const firstLink = recentExperimentsTable.locator("tbody tr a").first();

      // Check if there are any experiments
      const linkCount = await firstLink.count();
      if (linkCount === 0) {
        // No experiments, skip this test
        test.skip();
        return;
      }

      const className = await firstLink.textContent();

      // Click to navigate
      await firstLink.click();

      // Should be on detail page
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);
      await expect(
        page.getByRole("heading", { name: className || "" })
      ).toBeVisible();
    });

    test("should navigate to experiments list via 'View all' button", async ({
      page,
    }) => {
      await page.goto("/");

      // Click "View all" link
      await page.getByRole("link", { name: /View all/ }).click();

      // Should be on experiments list
      await expect(page).toHaveURL("/experiments");
      await expect(
        page.getByText("Browse and filter all Furu experiments")
      ).toBeVisible();
    });
  });

  test.describe("DAG to Detail Navigation", () => {
    test("should navigate to experiment detail from DAG node", async ({
      page,
    }) => {
      await page.goto("/dag");

      // Wait for DAG to load
      await expect(
        page.getByRole("heading", { name: "Experiment DAG" })
      ).toBeVisible();

      // Wait for the graph to render (nodes should appear)
      const dagContainer = page.locator(".react-flow");
      await expect(dagContainer).toBeVisible();

      // Click on a DAG node to select it
      const node = page
        .locator(".react-flow__node")
        .filter({ has: page.locator("a") })
        .first();
      const nodeCount = await node.count();

      if (nodeCount === 0) {
        // No nodes in DAG, skip
        test.skip();
        return;
      }

      await node.click({ position: { x: 10, y: 10 } });

      // A panel should appear with experiment links (React Flow Panel component)
      // The panel has class "react-flow__panel"
      const panel = page.locator(".react-flow__panel.m-4");
      await expect(panel).toBeVisible({ timeout: 5000 });

      // Click on an experiment link in the panel
      const experimentLink = panel.locator("a").first();
      const linkCount = await experimentLink.count();

      if (linkCount === 0) {
        test.skip();
        return;
      }

      await experimentLink.click();

      // Should navigate to detail page
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);
      await expect(page.getByRole("heading", { name: "Configuration" })).toBeVisible();
    });

    test("should navigate to experiment detail from DAG node inline links", async ({
      page,
    }) => {
      await page.goto("/dag");

      // Wait for DAG to load
      await expect(
        page.getByRole("heading", { name: "Experiment DAG" })
      ).toBeVisible();

      // The DAG nodes have inline experiment hash links - try clicking one
      const inlineLink = page.locator(".react-flow__node a").first();
      const linkCount = await inlineLink.count();

      if (linkCount === 0) {
        test.skip();
        return;
      }

      await inlineLink.click();

      // Should navigate to detail page
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);
    });
  });

  test.describe("Experiment Detail Relationship Navigation", () => {
    test("should navigate to parent experiment from detail page", async ({
      page,
    }) => {
      // TrainModel experiments have PrepareDataset as a parent dependency
      // Filter by class name to find a TrainModel experiment
      await page.goto("/experiments");
      await expect(
        page.getByText(/Showing \d+ of \d+ experiments/)
      ).toBeVisible();

      // Filter to TrainModel experiments which have parents (use namespace prefix)
      const namespaceInput = page.getByPlaceholder("Filter by namespace...");
      await namespaceInput.fill("my_project.pipelines.TrainModel");
      await expect(page.getByText(/Showing \d+ of [1-9]\d* experiments/)).toBeVisible();

      // Click on a TrainModel experiment
      await page.locator("table tbody tr").first().locator("a").first().click();
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);

      // TrainModel should have Parent Experiments section
      await expect(
        page.getByRole("heading", { name: "Parent Experiments" })
      ).toBeVisible();

      // Click on the parent experiment link
      const currentUrl = page.url();
      await page.getByText("View experiment").first().click();

      // Should navigate to the parent (PrepareDataset) detail page
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);
      expect(page.url()).not.toBe(currentUrl);
    });

    test("should navigate to child experiment from detail page", async ({
      page,
    }) => {
      // PrepareDataset experiments are parents to TrainModel/TrainTextModel
      // Filter by class name to find a PrepareDataset experiment
      await page.goto("/experiments");
      await expect(
        page.getByText(/Showing \d+ of \d+ experiments/)
      ).toBeVisible();

      // Filter to successful PrepareDataset experiments which have children
      const namespaceInput = page.getByPlaceholder("Filter by namespace...");
      await namespaceInput.fill("my_project.pipelines.PrepareDataset");

      const resultStatusSelect = page.getByRole("combobox").first();
      await resultStatusSelect.selectOption("success");

      const configInput = page.getByPlaceholder(
        "field.path=value (e.g., model.name=gpt-4)"
      );
      await configInput.fill("name=toy");

      await expect(page.getByText(/Showing \d+ of [1-9]\d* experiments/)).toBeVisible();

      await page.locator("table tbody tr").first().locator("a").first().click();
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);

      // PrepareDataset should have Child Experiments section
      await expect(
        page.getByRole("heading", { name: /Child Experiments/ })
      ).toBeVisible();

      // Click on a child experiment link
      const currentUrl = page.url();
      // The child links are inside the Child Experiments card
      const childCard = page.locator("text=Child Experiments").locator("xpath=ancestor::div[contains(@class, 'rounded-lg')]");
      await childCard.locator("a").first().click();

      // Should navigate to a child (TrainModel/TrainTextModel) detail page
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);
      expect(page.url()).not.toBe(currentUrl);
    });

    test("should navigate via dependency class link in configuration", async ({
      page,
    }) => {
      // TrainModel experiments have PrepareDataset as embedded config dependency
      await page.goto("/experiments");
      await expect(
        page.getByText(/Showing \d+ of \d+ experiments/)
      ).toBeVisible();

      // Filter to TrainModel experiments which have dependencies
      const namespaceInput = page.getByPlaceholder("Filter by namespace...");
      await namespaceInput.fill("my_project.pipelines.TrainModel");
      await expect(page.getByText(/Showing \d+ of [1-9]\d* experiments/)).toBeVisible();

      // Click on a TrainModel experiment
      await page.locator("table tbody tr").first().locator("a").first().click();
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);

      // Expand dependencies section
      const dependenciesButton = page.getByText("Dependencies (Embedded Config)");
      await expect(dependenciesButton).toBeVisible();
      await dependenciesButton.click();

      // Find the clickable class link (PrepareDataset) - it's an <a> tag with cyan color
      const classLink = page.locator("a.text-cyan-400").first();
      await expect(classLink).toBeVisible();

      const currentUrl = page.url();
      await classLink.click();

      // Should navigate to parent (PrepareDataset) detail page
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);
      expect(page.url()).not.toBe(currentUrl);
    });
  });

  test.describe("Direct URL Access", () => {
    test("should load experiment detail page directly via URL", async ({
      page,
    }) => {
      // First get a valid experiment URL
      await page.goto("/experiments");
      await expect(
        page.getByText(/Showing \d+ of \d+ experiments/)
      ).toBeVisible();

      // Get the link href
      const firstLink = page.locator("table tbody tr").first().locator("a").first();
      const href = await firstLink.getAttribute("href");

      if (!href) {
        test.skip();
        return;
      }

      // Navigate directly to that URL
      await page.goto(href);

      // Should show detail page, not list
      await expect(
        page.getByText("Browse and filter all Furu experiments")
      ).not.toBeVisible();
      await expect(page.getByRole("heading", { name: "Configuration" })).toBeVisible();
    });

    test("should show error for non-existent experiment", async ({ page }) => {
      // Navigate to a non-existent experiment
      await page.goto("/experiments/nonexistent.namespace/0000000000000000000");

      // Error should show quickly (no retries for 404)
      await expect(
        page.locator("h2", { hasText: "Experiment Not Found" })
      ).toBeVisible();

      // Should have a link back to experiments
      await page.getByRole("link", { name: "Back to experiments" }).click();
      await expect(page).toHaveURL("/experiments");
    });
  });

  test.describe("Browser Back/Forward Navigation", () => {
    test("should support browser back button after navigating to detail", async ({
      page,
    }) => {
      await page.goto("/experiments");
      await expect(
        page.getByText(/Showing \d+ of \d+ experiments/)
      ).toBeVisible();

      // Navigate to detail
      await page.locator("table tbody tr").first().locator("a").first().click();
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);

      // Press back
      await page.goBack();

      // Should be on experiments list
      await expect(page).toHaveURL("/experiments");
      await expect(
        page.getByText("Browse and filter all Furu experiments")
      ).toBeVisible();
    });

    test("should support browser forward button", async ({ page }) => {
      await page.goto("/experiments");
      await expect(
        page.getByText(/Showing \d+ of \d+ experiments/)
      ).toBeVisible();

      // Navigate to detail
      await page.locator("table tbody tr").first().locator("a").first().click();
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);

      // Go back then forward
      await page.goBack();
      await expect(page).toHaveURL("/experiments");

      await page.goForward();
      await expect(page).toHaveURL(/\/experiments\/.+\/.+/);
      await expect(page.getByRole("heading", { name: "Configuration" })).toBeVisible();
    });
  });
});
