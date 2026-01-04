import { test, expect } from "@playwright/test";

test.describe("API Endpoints", () => {
  test("health endpoint returns healthy", async ({ request }) => {
    const response = await request.get("/api/health");
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.status).toBe("healthy");
    expect(data.version).toBeDefined();
  });

  test("experiments endpoint returns list with generated data", async ({
    request,
  }) => {
    const response = await request.get("/api/experiments");
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.experiments).toBeDefined();
    expect(Array.isArray(data.experiments)).toBe(true);
    expect(typeof data.total).toBe("number");

    // Should have 10 experiments from our generated data
    expect(data.total).toBe(10);
  });

  test("experiments endpoint supports filtering by result status", async ({
    request,
  }) => {
    const response = await request.get(
      "/api/experiments?result_status=success"
    );
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.experiments).toBeDefined();
    // Should have 6 successful experiments
    expect(data.total).toBe(6);

    // All returned experiments should be successful
    for (const exp of data.experiments) {
      expect(exp.result_status).toBe("success");
    }
  });

  test("experiments endpoint supports filtering by attempt status", async ({
    request,
  }) => {
    const response = await request.get(
      "/api/experiments?attempt_status=running"
    );
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.experiments).toBeDefined();
    // Should have 1 running experiment
    expect(data.total).toBe(1);
    expect(data.experiments[0].attempt_status).toBe("running");
  });

  test("experiments endpoint supports pagination", async ({ request }) => {
    const response = await request.get("/api/experiments?limit=3&offset=0");
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.experiments.length).toBeLessThanOrEqual(3);
    expect(data.total).toBe(10); // Total count should still be 10

    // Get second page
    const response2 = await request.get("/api/experiments?limit=3&offset=3");
    const data2 = await response2.json();
    expect(data2.experiments.length).toBeLessThanOrEqual(3);

    // Experiments should be different
    if (data.experiments.length > 0 && data2.experiments.length > 0) {
      expect(data.experiments[0].huldra_hash).not.toBe(
        data2.experiments[0].huldra_hash
      );
    }
  });

  test("stats endpoint returns accurate statistics", async ({ request }) => {
    const response = await request.get("/api/stats");
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(typeof data.total).toBe("number");
    expect(typeof data.running_count).toBe("number");
    expect(typeof data.success_count).toBe("number");
    expect(typeof data.failed_count).toBe("number");
    expect(typeof data.queued_count).toBe("number");
    expect(Array.isArray(data.by_result_status)).toBe(true);

    // Verify counts match our generated data
    expect(data.total).toBe(10);
    expect(data.success_count).toBe(6);
    expect(data.running_count).toBe(1);
    expect(data.failed_count).toBe(1);
    expect(data.queued_count).toBe(1);
  });

  test("experiment detail endpoint returns full info", async ({ request }) => {
    // First, get list of experiments to get a valid namespace/hash
    const listResponse = await request.get(
      "/api/experiments?result_status=success&limit=1"
    );
    const listData = await listResponse.json();
    expect(listData.experiments.length).toBeGreaterThan(0);

    const experiment = listData.experiments[0];
    const { namespace, huldra_hash } = experiment;

    // Now fetch the detail
    const detailResponse = await request.get(
      `/api/experiments/${namespace}/${huldra_hash}`
    );
    expect(detailResponse.ok()).toBeTruthy();

    const detail = await detailResponse.json();
    expect(detail.namespace).toBe(namespace);
    expect(detail.huldra_hash).toBe(huldra_hash);
    expect(detail.class_name).toBeDefined();
    expect(detail.result_status).toBe("success");
    expect(detail.state).toBeDefined();
    expect(detail.metadata).toBeDefined();
    expect(detail.directory).toBeDefined();
  });

  test("nonexistent experiment returns 404", async ({ request }) => {
    const response = await request.get(
      "/api/experiments/nonexistent.namespace/fakehash123"
    );
    expect(response.status()).toBe(404);
  });

  test("experiments have proper metadata from Huldra", async ({ request }) => {
    const response = await request.get(
      "/api/experiments?result_status=success&limit=1"
    );
    const data = await response.json();
    expect(data.experiments.length).toBeGreaterThan(0);

    const experiment = data.experiments[0];

    // Fetch full detail to check metadata
    const detailResponse = await request.get(
      `/api/experiments/${experiment.namespace}/${experiment.huldra_hash}`
    );
    const detail = await detailResponse.json();

    // Check that metadata has expected fields from real Huldra objects
    expect(detail.metadata).toBeDefined();
    expect(detail.metadata.huldra_python_def).toBeDefined();
    expect(detail.metadata.huldra_obj).toBeDefined();
    expect(detail.metadata.huldra_hash).toBe(experiment.huldra_hash);
    expect(detail.metadata.git_commit).toBeDefined();
    expect(detail.metadata.hostname).toBeDefined();
    expect(detail.metadata.user).toBeDefined();
  });
});
