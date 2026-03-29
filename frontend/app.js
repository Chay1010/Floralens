/**
 * FloraLens — Frontend Application Logic
 * 
 * Handles image upload, drag & drop, API communication,
 * result rendering, background particles, and scroll effects.
 */

// ──── Configuration ────
const API_BASE = "http://localhost:8000";

// ──── DOM Elements ────
const elements = {
    uploadArea: document.getElementById("uploadArea"),
    uploadContent: document.getElementById("uploadContent"),
    fileInput: document.getElementById("fileInput"),
    browseBtn: document.getElementById("browseBtn"),
    previewContainer: document.getElementById("previewContainer"),
    previewImage: document.getElementById("previewImage"),
    changeImageBtn: document.getElementById("changeImageBtn"),
    scanBtn: document.getElementById("scanBtn"),
    scanLoader: document.getElementById("scanLoader"),
    resultsPanel: document.getElementById("resultsPanel"),
    resultsList: document.getElementById("resultsList"),
    inferenceBadge: document.getElementById("inferenceBadge"),
    navbar: document.getElementById("navbar"),
    navStatus: document.getElementById("navStatus"),
    bgParticles: document.getElementById("bgParticles"),
};

// ──── State ────
let currentFile = null;
let isScanning = false;

// ──── Initialize ────
document.addEventListener("DOMContentLoaded", () => {
    initParticles();
    initScrollEffects();
    initUpload();
    checkAPIHealth();
});

// ──── Background Particles ────
function initParticles() {
    const colors = [
        "rgba(16, 185, 129, 0.3)",
        "rgba(6, 182, 212, 0.2)",
        "rgba(132, 204, 22, 0.2)",
        "rgba(52, 211, 153, 0.25)",
        "rgba(20, 184, 166, 0.2)",
    ];

    for (let i = 0; i < 20; i++) {
        const particle = document.createElement("div");
        particle.className = "particle";
        const size = Math.random() * 6 + 3;
        const color = colors[Math.floor(Math.random() * colors.length)];
        const left = Math.random() * 100;
        const duration = Math.random() * 20 + 15;
        const delay = Math.random() * 20;

        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            left: ${left}%;
            animation-duration: ${duration}s;
            animation-delay: -${delay}s;
        `;
        elements.bgParticles.appendChild(particle);
    }
}

// ──── Scroll Effects ────
function initScrollEffects() {
    window.addEventListener("scroll", () => {
        const scrollY = window.scrollY;
        // Navbar background
        if (scrollY > 50) {
            elements.navbar.classList.add("scrolled");
        } else {
            elements.navbar.classList.remove("scrolled");
        }
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener("click", (e) => {
            e.preventDefault();
            const target = document.querySelector(anchor.getAttribute("href"));
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        });
    });
}

// ──── Upload Logic ────
function initUpload() {
    // Click to browse
    elements.browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        elements.fileInput.click();
    });

    elements.uploadArea.addEventListener("click", () => {
        if (!currentFile) {
            elements.fileInput.click();
        }
    });

    // File input change
    elements.fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });

    // Drag & drop
    elements.uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add("dragover");
    });

    elements.uploadArea.addEventListener("dragleave", () => {
        elements.uploadArea.classList.remove("dragover");
    });

    elements.uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            handleFile(file);
        }
    });

    // Change image
    elements.changeImageBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Scan button
    elements.scanBtn.addEventListener("click", () => {
        if (currentFile && !isScanning) {
            scanImage();
        }
    });
}

function handleFile(file) {
    // Validate
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
        showError("Please upload a JPEG, PNG, or WebP image.");
        return;
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showError("Image too large. Maximum size is 10MB.");
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.uploadContent.style.display = "none";
        elements.previewContainer.style.display = "block";
        elements.scanBtn.disabled = false;
        elements.resultsPanel.style.display = "none";
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    elements.fileInput.value = "";
    elements.uploadContent.style.display = "flex";
    elements.previewContainer.style.display = "none";
    elements.scanBtn.disabled = true;
    elements.resultsPanel.style.display = "none";
}

// ──── API Communication ────
async function checkAPIHealth() {
    const statusDot = elements.navStatus.querySelector(".status-dot");
    const statusText = elements.navStatus.querySelector(".status-text");

    try {
        const response = await fetch(`${API_BASE}/health`, { 
            signal: AbortSignal.timeout(5000) 
        });
        const data = await response.json();

        if (data.model_loaded) {
            statusDot.className = "status-dot online";
            statusText.textContent = "Model Ready";
        } else {
            statusDot.className = "status-dot";
            statusText.textContent = "Model Loading...";
        }
    } catch (err) {
        statusDot.className = "status-dot offline";
        statusText.textContent = "API Offline";
    }
}

async function scanImage() {
    if (!currentFile || isScanning) return;

    isScanning = true;
    elements.scanBtn.classList.add("loading");
    elements.resultsPanel.style.display = "none";

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        showError(err.message || "Failed to connect to the API. Make sure the backend is running.");
    } finally {
        isScanning = false;
        elements.scanBtn.classList.remove("loading");
    }
}

// ──── Results Rendering ────
function renderResults(data) {
    const { predictions, inference_time_ms } = data;

    // Inference badge
    elements.inferenceBadge.textContent = `⚡ ${inference_time_ms.toFixed(1)}ms`;

    // Build result items
    elements.resultsList.innerHTML = "";
    predictions.forEach((pred, index) => {
        const item = document.createElement("div");
        item.className = "result-item";
        item.style.animationDelay = `${index * 0.1}s`;

        const rankClass = index === 0 ? "rank-1" : index === 1 ? "rank-2" : index === 2 ? "rank-3" : "rank-other";
        const confidencePercent = (pred.confidence * 100).toFixed(1);
        const barWidth = Math.max(pred.confidence * 100, 2);

        item.innerHTML = `
            <div class="result-rank ${rankClass}">${index + 1}</div>
            <div class="result-info">
                <div class="result-name">${pred.class_name}</div>
                <div class="result-bar-container">
                    <div class="result-bar" style="width: 0%"></div>
                </div>
            </div>
            <div class="result-confidence">${confidencePercent}%</div>
        `;

        elements.resultsList.appendChild(item);

        // Animate bar
        requestAnimationFrame(() => {
            setTimeout(() => {
                const bar = item.querySelector(".result-bar");
                bar.style.width = `${barWidth}%`;
            }, 100 + index * 100);
        });
    });

    elements.resultsPanel.style.display = "block";

    // Scroll to results
    setTimeout(() => {
        elements.resultsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 300);
}

// ──── Error Handling ────
function showError(message) {
    // Create toast notification
    const toast = document.createElement("div");
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: rgba(244, 63, 94, 0.9);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-family: 'Outfit', sans-serif;
        font-size: 14px;
        font-weight: 500;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(244, 63, 94, 0.3);
        z-index: 1000;
        animation: slideUp 0.4s ease-out;
        max-width: 400px;
    `;
    toast.textContent = `⚠️ ${message}`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(20px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ──── Periodic Health Check ────
setInterval(checkAPIHealth, 30000);
