import { expect, test } from "@playwright/test";

test("browser registers through Platform and explores an interactive real Compute Studio", async ({ page }) => {
  const forbiddenApiRequests: string[] = [];
  const favoriteRequests: string[] = [];
  const gradientPreviewResponses: number[] = [];
  page.on("request", (request) => {
    const { pathname } = new URL(request.url());
    if (pathname.startsWith("/api/")) forbiddenApiRequests.push(request.url());
    if (pathname === "/platform/v1/me/favorites") favoriteRequests.push(request.url());
  });
  page.on("response", async (response) => {
    if (new URL(response.url()).pathname === "/platform/v1/studio/preview" && (await response.request().postDataJSON()).canonicalSpec.colorProgram) {
      gradientPreviewResponses.push(response.status());
    }
  });

  await page.goto("/register");
  await page.getByPlaceholder("Email").fill(`browser-${Date.now()}@example.test`);
  await page.getByPlaceholder("Password", { exact: true }).fill("browser-test-password");
  await page.getByPlaceholder("Confirm password").fill("browser-test-password");
  await page.getByRole("button", { name: "Create account" }).click();
  await page.waitForURL(/\/studio$/, { timeout: 30_000 });
  await expect(page.locator("main").getByRole("heading", { name: "Fractal Studio" })).toBeVisible();
  await expect(page.getByText("Drag pan · wheel zoom")).toBeVisible();
  await expect(page.getByAltText("Fractal preview")).toBeVisible({ timeout: 30_000 });

  await page.getByRole("button", { name: "Go", exact: true }).click();
  await expect(page.getByText("Recent history")).toBeVisible();
  await page.getByRole("button", { name: "Save view" }).click();
  await expect(page.getByText("View 1", { exact: false })).toBeVisible();
  await page.getByLabel("Custom gradient").check();
  await expect(page.locator('input[type="color"]')).toHaveCount(3);
  await expect.poll(() => gradientPreviewResponses).toContain(200);

  await page.route("**/platform/v1/me/assets?limit=48", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 300));
    await route.continue();
  });
  await page.getByRole("link", { name: "Library" }).click();
  await page.waitForURL(/\/assets$/, { timeout: 30_000 });
  await expect(page.getByRole("status")).toHaveText("Loading data…");

  await page.getByRole("link", { name: "Favorites" }).click();
  await page.waitForURL(/\/favorites$/, { timeout: 30_000 });
  await expect.poll(() => favoriteRequests).toHaveLength(1);
  await page.getByRole("link", { name: "Studio" }).click();
  await page.waitForURL(/\/studio$/, { timeout: 30_000 });
  await page.getByRole("link", { name: "Favorites" }).click();
  await page.waitForURL(/\/favorites$/, { timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Favorites" })).toBeVisible();
  expect(favoriteRequests).toHaveLength(1);

  expect(forbiddenApiRequests).toEqual([]);
});
