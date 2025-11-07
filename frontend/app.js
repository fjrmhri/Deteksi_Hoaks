const apiUrlInput = document.getElementById("apiUrl");
const saveApiBtn = document.getElementById("saveApiUrl");
const submitBtn = document.getElementById("submitBtn");
const newsText = document.getElementById("newsText");
const statusPanel = document.getElementById("statusPanel");
const resultPanel = document.getElementById("resultPanel");
const predictionWrapper = document.getElementById("prediction");

const STORAGE_KEY = "deteksi-hoaks-api-url";

function getApiBaseUrl() {
  return localStorage.getItem(STORAGE_KEY) || "";
}

function setApiBaseUrl(url) {
  if (url) {
    localStorage.setItem(STORAGE_KEY, url);
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function toggleStatus(message, type = "info") {
  statusPanel.textContent = message;
  statusPanel.classList.remove("hidden", "error", "visible");
  if (type === "error") {
    statusPanel.classList.add("error");
  }
  statusPanel.classList.add("visible");
}

function clearStatus() {
  statusPanel.classList.add("hidden");
  statusPanel.classList.remove("visible", "error");
}

function renderPrediction(result) {
  const isHoax = result.label.toLowerCase() === "hoaks";
  const emoji = isHoax ? "🔴" : "🟢";
  const label = isHoax ? "Hoaks" : "Bukan Hoaks";
  const card = document.createElement("div");
  card.className = `result-card ${isHoax ? "hoax" : "valid"}`;
  card.innerHTML = `
    <div class="label">${emoji} ${label}</div>
    <div class="score">Skor keyakinan: ${(result.skor * 100).toFixed(2)}%</div>
    <p class="text">${result.teks}</p>
  `;
  predictionWrapper.innerHTML = "";
  predictionWrapper.appendChild(card);
  resultPanel.classList.remove("hidden");
  resultPanel.classList.add("visible");
}

async function callApi(text) {
  const baseUrl = getApiBaseUrl();
  if (!baseUrl) {
    throw new Error("Silakan isi alamat API ngrok terlebih dahulu.");
  }

  const endpoint = `${baseUrl.replace(/\/$/, "")}/prediksi`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ teks: text }),
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    const detail = errorPayload.detail || response.statusText;
    throw new Error(`Permintaan gagal (${response.status}): ${detail}`);
  }

  return response.json();
}

async function handleSubmit() {
  clearStatus();
  resultPanel.classList.add("hidden");
  resultPanel.classList.remove("visible");

  const text = newsText.value.trim();
  if (!text) {
    toggleStatus("Masukkan teks berita terlebih dahulu.", "error");
    return;
  }

  toggleStatus("Menghubungi API FastAPI di Colab…");
  submitBtn.disabled = true;

  try {
    const payload = await callApi(text);
    if (!payload.hasil || !payload.hasil.length) {
      throw new Error("Respons API tidak sesuai harapan.");
    }
    renderPrediction(payload.hasil[0]);
    clearStatus();
  } catch (error) {
    toggleStatus(error.message, "error");
  } finally {
    submitBtn.disabled = false;
  }
}

saveApiBtn.addEventListener("click", () => {
  const url = apiUrlInput.value.trim();
  setApiBaseUrl(url);
  if (url) {
    toggleStatus("Alamat API tersimpan. Siap digunakan!");
  } else {
    toggleStatus("Alamat API dihapus. Masukkan URL baru sebelum memeriksa.");
  }
});

submitBtn.addEventListener("click", handleSubmit);

apiUrlInput.value = getApiBaseUrl();
if (apiUrlInput.value) {
  toggleStatus("Alamat API siap digunakan. Klik \"Periksa Sekarang\".");
}
