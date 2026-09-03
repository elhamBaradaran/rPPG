/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export - the whole dashboard is HTML/JS/JSON with no server, so it can be
  // hosted free on GitHub Pages or Vercel. Set BASE_PATH when serving from a
  // repository subpath, e.g. BASE_PATH=/rPPG npm run build.
  output: "export",
  basePath: process.env.BASE_PATH || "",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
