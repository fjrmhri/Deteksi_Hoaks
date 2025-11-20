const apiUrlInput = document.getElementById("apiUrl");
const saveApiBtn = document.getElementById("saveApiUrl");
const submitBtn = document.getElementById("submitBtn");
const newsText = document.getElementById("newsText");
const statusPanel = document.getElementById("statusPanel");
const resultPanel = document.getElementById("resultPanel");
const resultBadge = document.getElementById("resultBadge");
const resultText = document.getElementById("resultText");
const resultScore = document.getElementById("resultScore");

const STORAGE_KEY = "deteksi-hoaks-api-url";

const getApiBaseUrl = () => localStorage.getItem(STORAGE_KEY) || "http://localhost:8000";
const setApiBaseUrl = (url) => localStorage.setItem(STORAGE_KEY, url);

const setStatus = (message, type = "info") => {
  statusPanel.textContent = message;
  statusPanel.classList.remove("hidden", "error", "success");
  if (type === "error") statusPanel.classList.add("error");
  if (type === "success") statusPanel.classList.add("success");
};

const clearStatus = () => {
  statusPanel.textContent = "";
  statusPanel.classList.add("hidden");
  statusPanel.classList.remove("error", "success");
};

const renderResult = (item) => {
  const isHoax = item.label.toLowerCase().includes("hoax");
  resultBadge.textContent = isHoax ? "Hoaks" : "Bukan hoaks";
  resultBadge.style.background = isHoax ? "#fee2e2" : "#dcfce7";
  resultBadge.style.color = isHoax ? "#991b1b" : "#166534";
  resultText.textContent = item.teks;
  resultScore.textContent = `Confidence: ${(item.skor * 100).toFixed(2)}%`;
  resultPanel.classList.remove("hidden");
};

const validateApiUrl = (rawUrl) => {
  try {
    const pageIsHttps = window.location.protocol === "https:";

    // Allow relative proxy paths (e.g., /api/predict-hoax) so Vercel Functions
    // can forward to an HTTP backend without mixed-content issues.
    if (rawUrl.startsWith("/")) {
      return `${window.location.origin}${rawUrl.replace(/\/$/, "")}`;
    }

    const url = new URL(rawUrl);
    if (!url.protocol.startsWith("http")) {
      throw new Error("Gunakan protokol http atau https.");
    }
    const isLocalhost = ["localhost", "127.0.0.1", "::1"].includes(url.hostname);
    if (pageIsHttps && url.protocol === "http:" && !isLocalhost) {
      throw new Error(
        "Frontend berjalan di https sehingga backend http diblokir (mixed content). Pakai URL https dari tunneling (mis. Cloudflare/Ngrok) atau panggil proxy /api/predict-hoax."
      );
    }
    return url.toString().replace(/\/$/, "");
  } catch (err) {
    throw new Error(err.message || "URL backend tidak valid.");
  }
};

async function callApi(text) {
  const baseUrl = validateApiUrl(getApiBaseUrl());
  const endpoint = `${baseUrl}/predict-hoax`;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const message = detail.detail || response.statusText;
      throw new Error(`API error (${response.status}): ${message}`);
    }

    return response.json();
  } catch (err) {
    // Common cause: calling http backend from https frontend (blocked as mixed content)
    if (err instanceof TypeError) {
      throw new Error(
        "Gagal terhubung ke backend. Pastikan URL dapat diakses dan tidak diblok mixed content (https ke http)."
      );
    }
    throw err;
  }
}

async function handleSubmit() {
  clearStatus();
  resultPanel.classList.add("hidden");
  const text = newsText.value.trim();
  if (!text) {
    setStatus("Masukkan teks berita terlebih dahulu.", "error");
    return;
  }

  setStatus("Memproses di backend...", "info");
  submitBtn.disabled = true;
  saveApiBtn.disabled = true;

  try {
    const payload = await callApi(text);
    if (!payload.label || payload.score === undefined) {
      throw new Error("Respons API tidak sesuai.");
    }
    renderResult({
      label: payload.label,
      skor: payload.score,
      teks: text,
    });
    setStatus("Berhasil memuat prediksi.", "success");
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    submitBtn.disabled = false;
    saveApiBtn.disabled = false;
  }
}

saveApiBtn.addEventListener("click", () => {
  const rawUrl = apiUrlInput.value.trim() || "http://localhost:8000";
  try {
    const cleanUrl = validateApiUrl(rawUrl);
    setApiBaseUrl(cleanUrl);
    setStatus("URL backend tersimpan.", "success");
  } catch (err) {
    setStatus(err.message, "error");
  }
});

submitBtn.addEventListener("click", handleSubmit);

// Initialise form with stored URL
apiUrlInput.value = getApiBaseUrl();
try {
  setStatus("Siap menggunakan backend: " + validateApiUrl(apiUrlInput.value), "success");
} catch (err) {
  setStatus(err.message, "error");
}
