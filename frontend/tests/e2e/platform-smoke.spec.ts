import { expect, test } from "@playwright/test";

test("browser registers through Platform and renders a real Compute preview", async ({ page }) => {
  const forbiddenApiRequests: string[] = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.startsWith("/api/")) forbiddenApiRequests.push(request.url());
  });

  await page.goto("/register");
  await page.getByPlaceholder("Email").fill(`browser-${Date.now()}@example.test`);
  await page.getByPlaceholder("Password", { exact: true }).fill("browser-test-password");
  await page.getByPlaceholder("Confirm password").fill("browser-test-password");
  await page.getByRole("button", { name: "Create account" }).click();
  await page.waitForURL(/\/studio$/, { timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Platform Studio" })).toBeVisible();

  await page.getByRole("button", { name: "Preview" }).click();
  await expect(page.getByAltText("Fractal preview")).toBeVisible({ timeout: 30_000 });
  expect(forbiddenApiRequests).toEqual([]);
});
