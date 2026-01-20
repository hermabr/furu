/**
 * Tests for Zod schema validation.
 * These tests ensure that mock data has the same shape as the OpenAPI schema.
 */
import { expect, test, describe } from "bun:test";
import {
  healthCheckApiHealthGetResponse,
  listExperimentsApiExperimentsGetResponse,
  getExperimentApiExperimentsNamespaceFuruHashGetResponse,
  dashboardStatsApiStatsGetResponse,
} from "./api/zod/schemas";

// Mock data that mirrors the Python backend's responses
const mockData = {
  health: {
    status: "healthy",
    version: "0.1.0",
  },
    experimentSummary: {
      namespace: "my_project.pipelines.TrainModel",
      furu_hash: "abc123def456",
      class_name: "TrainModel",
      result_status: "success",
      attempt_status: "success",
      attempt_number: 1,
      updated_at: "2025-01-01T12:00:00+00:00",
      started_at: "2025-01-01T11:00:00+00:00",
      migration_kind: "copied",
      migration_policy: "copy",
      migrated_at: "2025-01-02T00:00:00+00:00",
      overwritten_at: null,
      migration_origin: "tests",
      migration_note: "migration test",
      from_namespace: "my_project.pipelines.TrainModel",
      from_hash: "orig123",
      to_namespace: "my_project.pipelines.TrainModel",
      to_hash: "abc123def456",
      original_result_status: "success",
    },
    experimentList: {
      experiments: [
        {
          namespace: "my_project.pipelines.TrainModel",
          furu_hash: "abc123def456",
          class_name: "TrainModel",
          result_status: "success",
          attempt_status: "success",
          attempt_number: 1,
          updated_at: "2025-01-01T12:00:00+00:00",
          started_at: "2025-01-01T11:00:00+00:00",
          migration_kind: "moved",
          migration_policy: "move",
          migrated_at: "2025-01-02T00:00:00+00:00",
          overwritten_at: null,
          migration_origin: "tests",
          migration_note: "migration test",
          from_namespace: "my_project.pipelines.TrainModel",
          from_hash: "orig123",
          to_namespace: "my_project.pipelines.TrainModel",
          to_hash: "abc123def456",
          original_result_status: "success",
        },
        {
          namespace: "my_project.pipelines.TrainModel",
          furu_hash: "def789ghi012",
          class_name: "TrainModel",
          result_status: "success",
          attempt_status: "success",
          attempt_number: 1,
          updated_at: "2025-01-02T12:00:00+00:00",
          started_at: "2025-01-02T11:00:00+00:00",
          migration_kind: "copied",
          migration_policy: "copy",
          migrated_at: "2025-01-03T00:00:00+00:00",
          overwritten_at: null,
          migration_origin: "tests",
          migration_note: "migration test",
          from_namespace: "my_project.pipelines.TrainModel",
          from_hash: "orig456",
          to_namespace: "my_project.pipelines.TrainModel",
          to_hash: "def789ghi012",
          original_result_status: "success",
        },
      ],
      total: 2,
    },
    experimentDetail: {
      namespace: "my_project.pipelines.TrainModel",
      furu_hash: "def789ghi012",
      class_name: "TrainModel",
      result_status: "success",
      attempt_status: "success",
      attempt_number: 1,
      updated_at: "2025-01-02T12:00:00+00:00",
      started_at: "2025-01-02T11:00:00+00:00",
      backend: "local",
      hostname: "test-host",
      user: "testuser",
      directory: "/data/my_project/pipelines/TrainModel/def789ghi012",
      state: {
        schema_version: 1,
        result: { status: "success", created_at: "2025-01-02T12:00:00+00:00" },
        attempt: null,
        updated_at: "2025-01-02T12:00:00+00:00",
      },
      metadata: {
        furu_python_def: "TrainModel(lr=0.001)",
        git_commit: "abc123",
        git_branch: "main",
        hostname: "test-host",
        user: "testuser",
      },
      attempt: {
        id: "attempt-def789",
        number: 1,
        backend: "local",
        status: "success",
        started_at: "2025-01-02T11:00:00+00:00",
        heartbeat_at: "2025-01-02T11:30:00+00:00",
        lease_duration_sec: 7200,
        lease_expires_at: "2025-01-02T13:00:00+00:00",
        owner: {
          pid: 12345,
          host: "test-host",
          user: "testuser",
        },
        ended_at: "2025-01-02T12:00:00+00:00",
      },
      migration_kind: "copied",
      migration_policy: "copy",
      migrated_at: "2025-01-03T00:00:00+00:00",
      overwritten_at: null,
      migration_origin: "tests",
      migration_note: "migration test",
      from_namespace: "my_project.pipelines.TrainModel",
      from_hash: "orig456",
      to_namespace: "my_project.pipelines.TrainModel",
      to_hash: "def789ghi012",
      original_result_status: "success",
      original_namespace: "my_project.pipelines.TrainModel",
      original_hash: "orig456",
      alias_namespaces: ["my_project.pipelines.TrainModel"],
      alias_hashes: ["alias456"],
    },

  stats: {
    total: 10,
    by_result_status: [
      { status: "success", count: 5 },
      { status: "failed", count: 2 },
      { status: "incomplete", count: 2 },
      { status: "absent", count: 1 },
    ],
    by_attempt_status: [
      { status: "success", count: 5 },
      { status: "running", count: 2 },
      { status: "failed", count: 2 },
      { status: "queued", count: 1 },
    ],
    running_count: 2,
    queued_count: 1,
    failed_count: 2,
    success_count: 5,
  },
};

describe("Health Check Schema", () => {
  test("valid health response passes schema", () => {
    const result = healthCheckApiHealthGetResponse.safeParse(mockData.health);
    expect(result.success).toBe(true);
  });

  test("invalid health response fails schema", () => {
    const invalidData = { status: 123 }; // status should be string
    const result = healthCheckApiHealthGetResponse.safeParse(invalidData);
    expect(result.success).toBe(false);
  });

  test("missing version fails schema", () => {
    const invalidData = { status: "healthy" };
    const result = healthCheckApiHealthGetResponse.safeParse(invalidData);
    expect(result.success).toBe(false);
  });
});

describe("Experiment List Schema", () => {
  test("valid experiment list passes schema", () => {
    const result = listExperimentsApiExperimentsGetResponse.safeParse(
      mockData.experimentList
    );
    expect(result.success).toBe(true);
  });

  test("empty experiment list passes schema", () => {
    const emptyList = { experiments: [], total: 0 };
    const result =
      listExperimentsApiExperimentsGetResponse.safeParse(emptyList);
    expect(result.success).toBe(true);
  });

  test("missing total fails schema", () => {
    const invalidList = { experiments: [] };
    const result =
      listExperimentsApiExperimentsGetResponse.safeParse(invalidList);
    expect(result.success).toBe(false);
  });

  test("experiment with missing namespace fails schema", () => {
    const invalidList = {
      experiments: [
        {
          furu_hash: "abc123",
          class_name: "Test",
          result_status: "success",
        },
      ],
      total: 1,
    };
    const result =
      listExperimentsApiExperimentsGetResponse.safeParse(invalidList);
    expect(result.success).toBe(false);
  });
});

describe("Experiment Detail Schema", () => {
  test("valid experiment detail passes schema", () => {
    const result = getExperimentApiExperimentsNamespaceFuruHashGetResponse.safeParse(
      mockData.experimentDetail
    );
    expect(result.success).toBe(true);
  });

  test("experiment detail with null metadata passes schema", () => {
    const detailWithNullMetadata = {
      ...mockData.experimentDetail,
      metadata: null,
    };
    const result = getExperimentApiExperimentsNamespaceFuruHashGetResponse.safeParse(
      detailWithNullMetadata
    );
    expect(result.success).toBe(true);
  });

  test("experiment detail with null attempt passes schema", () => {
    const detailWithNullAttempt = {
      ...mockData.experimentDetail,
      attempt: null,
    };
    const result = getExperimentApiExperimentsNamespaceFuruHashGetResponse.safeParse(
      detailWithNullAttempt
    );
    expect(result.success).toBe(true);
  });

  test("experiment detail missing directory fails schema", () => {
    const { directory: _, ...invalidDetail } = mockData.experimentDetail;
    const result = getExperimentApiExperimentsNamespaceFuruHashGetResponse.safeParse(
      invalidDetail
    );
    expect(result.success).toBe(false);
  });
});

describe("Dashboard Stats Schema", () => {
  test("valid stats passes schema", () => {
    const result = dashboardStatsApiStatsGetResponse.safeParse(mockData.stats);
    expect(result.success).toBe(true);
  });

  test("stats with empty status arrays passes schema", () => {
    const emptyStats = {
      total: 0,
      by_result_status: [],
      by_attempt_status: [],
      running_count: 0,
      queued_count: 0,
      failed_count: 0,
      success_count: 0,
    };
    const result = dashboardStatsApiStatsGetResponse.safeParse(emptyStats);
    expect(result.success).toBe(true);
  });

  test("stats missing total fails schema", () => {
    const { total: _, ...invalidStats } = mockData.stats;
    const result = dashboardStatsApiStatsGetResponse.safeParse(invalidStats);
    expect(result.success).toBe(false);
  });

  test("stats with invalid status count fails schema", () => {
    const invalidStats = {
      ...mockData.stats,
      by_result_status: [{ status: "success" }], // missing count
    };
    const result = dashboardStatsApiStatsGetResponse.safeParse(invalidStats);
    expect(result.success).toBe(false);
  });
});

describe("Schema type inference", () => {
  test("inferred types match expected structure", () => {
    const health = healthCheckApiHealthGetResponse.parse(mockData.health);
    expect(typeof health.status).toBe("string");
    expect(typeof health.version).toBe("string");

    const stats = dashboardStatsApiStatsGetResponse.parse(mockData.stats);
    expect(typeof stats.total).toBe("number");
    expect(typeof stats.running_count).toBe("number");
    expect(Array.isArray(stats.by_result_status)).toBe(true);
  });
});



