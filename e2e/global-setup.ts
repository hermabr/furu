/**
 * Global setup for Playwright e2e tests.
 * 
 * This script runs before all tests to generate test data using the
 * Python data generation script. It creates realistic Huldra experiments
 * with various states and dependencies.
 */

import { execSync } from "child_process";
import * as path from "path";

async function globalSetup() {
  console.log("🔧 Generating test data for e2e tests...");

  const projectRoot = path.resolve(__dirname, "..");
  const e2eDir = __dirname;

  try {
    // Run the data generation script with --clean to ensure fresh data
    execSync(`uv run python ${path.join(e2eDir, "generate_data.py")} --clean`, {
      cwd: projectRoot,
      stdio: "inherit",
      env: {
        ...process.env,
        // Ensure we use the project's data-huldra directory
        HULDRA_PATH: path.join(projectRoot, "data-huldra"),
      },
    });
    console.log("✅ Test data generated successfully");
  } catch (error) {
    console.error("❌ Failed to generate test data:", error);
    throw error;
  }
}

export default globalSetup;
