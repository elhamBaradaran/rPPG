/** @type {import('next').NextConfig} */

// GitHub Pages serves a project site from /<repo>, so every asset URL needs that prefix.
// Vercel and a local dev server serve from the root, where it must be empty.
// NEXT_PUBLIC_BASE_PATH is the same value, exposed to client code (see lib/data.ts).
const basePath = process.env.BASE_PATH || "";

const nextConfig = {
  // Static export - the whole dashboard is HTML, JS and JSON with no server, so it hosts
  // free on GitHub Pages or Vercel.
  output: "export",
  basePath,
  images: { unoptimized: true },
  trailingSlash: true,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
