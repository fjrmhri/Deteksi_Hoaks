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

async function callApi(text) {
  const baseUrl = getApiBaseUrl().replace(/\/$/, "");
  const endpoint = `${baseUrl}/predict-hoax`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ teks: text }),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = detail.detail || response.statusText;
    throw new Error(`API error (${response.status}): ${message}`);
  }

  return response.json();
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
    if (!payload.hasil || !payload.hasil.length) {
      throw new Error("Respons API tidak sesuai.");
    }
    renderResult(payload.hasil[0]);
    setStatus("Berhasil memuat prediksi.", "success");
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    submitBtn.disabled = false;
    saveApiBtn.disabled = false;
  }
}

saveApiBtn.addEventListener("click", () => {
  const url = apiUrlInput.value.trim() || "http://localhost:8000";
  setApiBaseUrl(url);
  setStatus("URL backend tersimpan.", "success");
});

submitBtn.addEventListener("click", handleSubmit);

// Initialise form with stored URL
apiUrlInput.value = getApiBaseUrl();
setStatus("Siap menggunakan backend: " + apiUrlInput.value, "success");
