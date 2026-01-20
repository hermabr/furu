/**
 * Global setup for Playwright e2e tests.
 *
 * This script runs before all tests to generate test data using the
 * Python data generation script. It creates realistic Furu experiments
 * with various states and dependencies in a temporary data directory.
 */

import { execSync } from "child_process";
import * as path from "path";

async function globalSetup() {
  const projectRoot = path.resolve(__dirname, "..");
  const e2eDir = __dirname;
  const dataDir = process.env.FURU_E2E_DATA_DIR ?? process.env.FURU_PATH;

  if (!dataDir) {
    throw new Error("FURU_E2E_DATA_DIR must be set for e2e tests");
  }

  console.log(`🔧 Generating test data for e2e tests in ${dataDir}...`);

  execSync(
    `uv run python ${path.join(e2eDir, "generate_data.py")} --data-dir ${dataDir}`,
    {
      cwd: projectRoot,
      stdio: "inherit",
      env: {
        ...process.env,
        FURU_PATH: dataDir,
      },
    }
  );
  console.log("✅ Test data generated successfully");
}

export default globalSetup;
