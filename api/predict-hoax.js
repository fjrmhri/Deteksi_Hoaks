const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL || "http://localhost:8000";

const normalizeBase = (raw) => raw.replace(/\/$/, "");

module.exports = async (req, res) => {
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ detail: "Method not allowed" });
    return;
  }

  const baseUrl = normalizeBase(BACKEND_BASE_URL);
  const targetUrl = `${baseUrl}/predict-hoax`;

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
    const upstream = await fetch(targetUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      res.status(upstream.status).json({ detail: data?.detail || "Upstream error" });
      return;
    }

    res.status(200).json(data);
  } catch (err) {
    res.status(500).json({ detail: `Proxy error: ${err.message}` });
  }
};
