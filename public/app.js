// Ambil elemen DOM utama
const submitBtn = document.getElementById("submitBtn");
const submitLabel = document.getElementById("submitLabel");
const submitSpinner = document.getElementById("submitSpinner");
const newsText = document.getElementById("newsText");
const statusPanel = document.getElementById("statusPanel");
const resultPanel = document.getElementById("resultPanel");
const resultBadge = document.getElementById("resultBadge");
const resultText = document.getElementById("resultText");
const resultScore = document.getElementById("resultScore");
const resultDecision = document.getElementById("resultDecision");
const resultRiskExplanation = document.getElementById("resultRiskExplanation");

const copyBtn = document.getElementById("copyBtn");
const shareBtn = document.getElementById("shareBtn");

let lastResultShareText = "";

// =========================
// Konfigurasi API base URL
// =========================

// API base URL di-inject dari Vercel via window.__ENV__
// Jika placeholder belum diganti (masih "%NEXT_PUBLIC_API_BASE_URL%"),
// kita anggap belum terkonfigurasi.
const rawEnv =
  window.__ENV && typeof window.__ENV.API_BASE_URL === "string"
    ? window.__ENV.API_BASE_URL.trim()
    : "";

const apiBaseUrl =
  rawEnv && rawEnv !== "%NEXT_PUBLIC_API_BASE_URL%"
    ? rawEnv.replace(/\/+$/, "")
    : "";

// =========================
// Helper: Status UI
// =========================

const setStatus = (message, type = "info") => {
  if (!statusPanel) return;
  statusPanel.textContent = message;
  statusPanel.classList.remove("hidden", "error", "success");
  if (type === "error") statusPanel.classList.add("error");
  if (type === "success") statusPanel.classList.add("success");
};

const clearStatus = () => {
  if (!statusPanel) return;
  statusPanel.textContent = "";
  statusPanel.classList.add("hidden");
  statusPanel.classList.remove("error", "success");
};

// =========================
// Spinner & tombol
// =========================

const setLoading = (isLoading) => {
  if (!submitBtn || !submitSpinner || !submitLabel) return;
  submitBtn.disabled = isLoading;
  if (isLoading) {
    submitSpinner.classList.remove("hidden");
    submitLabel.textContent = "Memproses...";
  } else {
    submitSpinner.classList.add("hidden");
    submitLabel.textContent = "Periksa sekarang";
  }
};

// =========================
// Cek /health backend
// =========================

async function verifyBackend(url) {
  const base = url.replace(/\/+$/, "");
  const endpoint = `${base}/health`;

  try {
    const res = await fetch(endpoint, { method: "GET" });
    if (!res.ok) {
      throw new Error(`Backend merespons kode ${res.status}`);
    }
    return true;
  } catch (err) {
    console.error("Gagal memverifikasi backend:", err);
    // Jangan tampilkan "Failed to fetch" mentah-mentah ke user
    if (err instanceof TypeError) {
      throw new Error(
        "Gagal menghubungi backend. Pastikan URL NEXT_PUBLIC_API_BASE_URL sudah benar (https) dan Space aktif."
      );
    }
    const message =
      err instanceof Error ? err.message : "Tidak dapat terhubung ke backend.";
    throw new Error(message);
  }
}

// =========================
// Map risk level → badge
// =========================

const mapRiskToBadge = (riskLevel) => {
  const level = String(riskLevel || "")
    .toLowerCase()
    .trim();
  if (level === "high") {
    return {
      text: "Hoaks – risiko tinggi",
      className: "badge badge--high",
    };
  }
  if (level === "medium") {
    return {
      text: "Perlu dicek (curiga)",
      className: "badge badge--medium",
    };
  }
  if (level === "low") {
    return {
      text: "Bukan hoaks (cenderung valid)",
      className: "badge badge--low",
    };
  }
  return {
    text: "Level risiko tidak diketahui",
    className: "badge",
  };
};

// =========================
// Render hasil prediksi
// =========================

const renderResult = (prediction, originalText) => {
  if (
    !resultPanel ||
    !resultBadge ||
    !resultText ||
    !resultScore ||
    !resultDecision ||
    !resultRiskExplanation
  )
    return;

  const {
    label,
    score,
    probabilities,
    hoaxProbability,
    riskLevel,
    riskExplanation,
  } = prediction;

  const normalizedLabel = String(label || "")
    .toLowerCase()
    .trim();
  const isHoax =
    normalizedLabel === "hoax" ||
    normalizedLabel === "1" ||
    normalizedLabel.includes("hoax");

  // Badge berdasarkan risk level
  const badgeInfo = mapRiskToBadge(riskLevel);
  resultBadge.textContent = badgeInfo.text;
  resultBadge.className = badgeInfo.className;

  // Prediksi model (label argmax)
  if (isHoax) {
    resultDecision.textContent = "Prediksi model: Hoaks";
  } else if (
    normalizedLabel === "not_hoax" ||
    normalizedLabel === "non_hoax" ||
    normalizedLabel === "0"
  ) {
    resultDecision.textContent = "Prediksi model: Bukan hoaks";
  } else {
    resultDecision.textContent = `Prediksi model: ${
      label ?? "Tidak diketahui"
    }`;
  }

  // Penjelasan risiko dari backend
  resultRiskExplanation.textContent = riskExplanation || "";

  // Skor & probabilitas
  let pHoax = null;
  let pNotHoax = null;

  if (typeof hoaxProbability === "number") {
    pHoax = hoaxProbability;
  } else if (probabilities && typeof probabilities === "object") {
    if (typeof probabilities.hoax === "number") {
      pHoax = probabilities.hoax;
    } else if (typeof probabilities.Hoax === "number") {
      pHoax = probabilities.Hoax;
    }
  }

  if (probabilities && typeof probabilities === "object") {
    if (typeof probabilities.not_hoax === "number") {
      pNotHoax = probabilities.not_hoax;
    } else if (typeof probabilities["not hoax"] === "number") {
      pNotHoax = probabilities["not hoax"];
    }
  }

  let scoreText = `Confidence: ${(score * 100).toFixed(2)}%`;

  const parts = [];
  if (typeof pHoax === "number") {
    parts.push(`P(hoaks): ${(pHoax * 100).toFixed(2)}%`);
  }
  if (typeof pNotHoax === "number") {
    parts.push(`P(bukan hoaks): ${(pNotHoax * 100).toFixed(2)}%`);
  }
  if (parts.length > 0) {
    scoreText = parts.join(" • ");
  }

  resultScore.textContent = scoreText;
  resultText.textContent = originalText;

  // Siapkan teks untuk Copy / Share
  lastResultShareText =
    `Hasil Deteksi Hoaks\n\n` +
    `${resultDecision.textContent}\n` +
    `${resultBadge.textContent}\n` +
    `${scoreText}\n\n` +
    `Penjelasan: ${riskExplanation || "-"}\n\n` +
    `Teks berita:\n${originalText}`;

  resultPanel.classList.remove("hidden");
};

// =========================
// Panggil API FastAPI
// =========================

async function callApi(text) {
  if (!apiBaseUrl) {
    throw new Error(
      "NEXT_PUBLIC_API_BASE_URL belum terkonfigurasi. Set env di Vercel terlebih dahulu."
    );
  }

  const base = apiBaseUrl.replace(/\/+$/, "");
  const endpoint = `${base}/predict`;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json().catch(() => null);
    if (!response.ok || !data) {
      const message =
        (data && data.detail) ||
        response.statusText ||
        "API tidak merespons dengan benar.";
      throw new Error(`API error (${response.status}): ${message}`);
    }

    return data;
  } catch (err) {
    console.error("Gagal memanggil API hoaks:", err);
    if (err instanceof TypeError) {
      // Di sinilah "Failed to fetch" biasanya muncul → kita ganti dengan pesan ramah
      throw new Error(
        "Gagal terhubung ke backend. Pastikan URL API (HTTPS) benar dan Space Hugging Face sedang aktif."
      );
    }
    const message =
      err instanceof Error ? err.message : "Terjadi kesalahan tak terduga.";
    throw new Error(message);
  }
}

const extractPrediction = (payload) => {
  // Ekspektasi respons FastAPI:
  // {
  //   "label": "hoax" | "not_hoax",
  //   "score": 0.xx,
  //   "probabilities": {...},
  //   "hoax_probability": 0.xx,
  //   "risk_level": "high" | "medium" | "low",
  //   "risk_explanation": "..."
  // }
  if (
    !payload ||
    typeof payload.label !== "string" ||
    typeof payload.score !== "number"
  ) {
    throw new Error("Respons API tidak sesuai format yang diharapkan.");
  }

  return {
    label: payload.label,
    score: payload.score,
    probabilities: payload.probabilities || null,
    hoaxProbability:
      typeof payload.hoax_probability === "number"
        ? payload.hoax_probability
        : null,
    riskLevel: payload.risk_level || "unknown",
    riskExplanation: payload.risk_explanation || "",
  };
};

// =========================
// Copy & Share helpers
// =========================

async function handleCopy() {
  if (!lastResultShareText) {
    setStatus("Belum ada hasil untuk dicopy.", "error");
    return;
  }
  try {
    await navigator.clipboard.writeText(lastResultShareText);
    setStatus("Hasil berhasil disalin ke clipboard.", "success");
  } catch (err) {
    console.error("Gagal copy:", err);
    setStatus("Gagal menyalin ke clipboard.", "error");
  }
}

async function handleShare() {
  if (!lastResultShareText) {
    setStatus("Belum ada hasil untuk dibagikan.", "error");
    return;
  }

  if (navigator.share) {
    try {
      await navigator.share({
        title: "Hasil Deteksi Hoaks",
        text: lastResultShareText,
      });
    } catch (err) {
      console.error("Share dibatalkan / gagal:", err);
    }
  } else {
    // Fallback: copy saja
    await handleCopy();
  }
}

// =========================
// Handler submit
// =========================

async function handleSubmit() {
  clearStatus();
  if (resultPanel) resultPanel.classList.add("hidden");

  const text = newsText ? newsText.value.trim() : "";
  if (!text) {
    setStatus("Masukkan teks berita terlebih dahulu.", "error");
    return;
  }

  setStatus("Memproses di backend...", "info");
  setLoading(true);

  try {
    const payload = await callApi(text);
    const prediction = extractPrediction(payload);
    renderResult(prediction, text);
    setStatus("Berhasil memuat prediksi.", "success");
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Terjadi kesalahan saat memproses.";
    setStatus(message, "error");
  } finally {
    setLoading(false);
  }
}

// Event klik tombol
if (submitBtn) {
  submitBtn.addEventListener("click", handleSubmit);
}

// Event Ctrl+Enter di textarea untuk submit cepat
if (newsText) {
  newsText.addEventListener("keydown", (e) => {
    if (e.ctrlKey && (e.key === "Enter" || e.key === "NumpadEnter")) {
      e.preventDefault();
      handleSubmit();
    }
  });
}

// Copy & Share events
if (copyBtn) copyBtn.addEventListener("click", handleCopy);
if (shareBtn) shareBtn.addEventListener("click", handleShare);

// =========================
// Inisialisasi awal
// =========================

if (!apiBaseUrl) {
  setStatus(
    "Konfigurasi API tidak ditemukan. Set NEXT_PUBLIC_API_BASE_URL di Vercel (gunakan URL HTTPS Space Hugging Face).",
    "error"
  );
  if (submitBtn) submitBtn.disabled = true;
} else {
  setStatus(`Menggunakan backend: ${apiBaseUrl}`, "info");
  verifyBackend(apiBaseUrl)
    .then(() => setStatus(`Tersambung ke backend: ${apiBaseUrl}`, "success"))
    .catch((err) => setStatus(err.message, "error"));
}
