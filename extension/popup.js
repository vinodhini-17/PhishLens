/**
 * PhishLens - Chrome Extension Popup Logic
 * popup.js
 *
 * 1. Queries the currently active Chrome tab to retrieve its URL.
 * 2. Displays the active URL in popup.html.
 * 3. Sends a POST request via fetch() to http://127.0.0.1:5000/predict.
 * 4. Dynamically renders the model prediction, confidence score, and top 3 SHAP risk factors:
 *    - "Safe": Green aesthetic, "Legitimate Website", confidence meter, and trust/safety factors.
 *    - "Phishing": Red warning aesthetic, "Phishing Detected!", and bulleted SHAP risk factors.
 * 5. Preserves the Adaptive Memory feedback loop via the 'Report Mistake' button.
 */

document.addEventListener("DOMContentLoaded", () => {
  // UI Elements
  const currentUrlEl = document.getElementById("currentUrl");
  const statusBlock = document.getElementById("statusBlock");
  const iconSvg = document.getElementById("iconSvg");
  const statusTitle = document.getElementById("statusTitle");
  const statusDescription = document.getElementById("statusDescription");
  const confidenceMeter = document.getElementById("confidenceMeter");
  
  // SHAP Risk Factors Elements
  const riskFactorsCard = document.getElementById("riskFactorsCard");
  const riskCardTitle = document.getElementById("riskCardTitle");
  const riskFactorsList = document.getElementById("riskFactorsList");

  // Feedback Elements
  const reportMistakeBtn = document.getElementById("reportMistakeBtn");
  const feedbackMessage = document.getElementById("feedbackMessage");
  const backendStatusText = document.getElementById("backendStatusText");

  const API_ENDPOINT = "http://127.0.0.1:5000/predict";
  let activeTabUrl = "";
  let currentVerdict = null;

  // --------------------------------------------------------------------------
  // 1. Query Active Chrome Tab URL
  // --------------------------------------------------------------------------
  if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.query) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs.length > 0 && tabs[0].url) {
        activeTabUrl = tabs[0].url;
        handleUrlInspection(activeTabUrl);
      } else {
        displayFallbackUrl("No active webpage detected");
      }
    });
  } else {
    // Development / Browser preview fallback
    activeTabUrl = "https://www.drishtiias.com/upsc-syllabus";
    handleUrlInspection(activeTabUrl);
  }

  function handleUrlInspection(url) {
    currentUrlEl.textContent = url;
    currentUrlEl.title = url;

    // Check for internal browser URLs
    if (url.startsWith("chrome://") || url.startsWith("chrome-extension://") || url.startsWith("about:") || url.startsWith("edge://")) {
      setNeutralState(
        "Internal Page",
        "Chrome internal and system pages are secure by default.",
        "System Protected"
      );
      if (riskFactorsCard) riskFactorsCard.style.display = "none";
      return;
    }

    // Set initial scanning state and trigger API prediction
    setScanningState();
    fetchPredictionFromApi(url);
  }

  function displayFallbackUrl(msg) {
    currentUrlEl.textContent = msg;
    setNeutralState("Inactive", "Open a valid website tab to scan.", "Standby");
    if (riskFactorsCard) riskFactorsCard.style.display = "none";
  }

  // --------------------------------------------------------------------------
  // 2. Fetch Prediction & SHAP Explainability from Python Backend (app.py)
  // --------------------------------------------------------------------------
  async function fetchPredictionFromApi(url) {
    try {
      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ url: url })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      currentVerdict = data;

      // Update backend status footer
      if (backendStatusText) {
        backendStatusText.textContent = "AI Engine: Connected (Local API)";
      }

      const isPhishing = (String(data.prediction).toLowerCase() === "phishing");
      const confidence = typeof data.confidence_score === "number" ? data.confidence_score : 100.0;
      const riskFactors = Array.isArray(data.risk_factors) ? data.risk_factors : [];

      // Check if domain was verified via Allowlist
      const isAllowlisted = riskFactors.some(f => f.toLowerCase().includes("allowlist") || f.toLowerCase().includes("trusted domain"));

      if (isPhishing) {
        // Red warning aesthetic
        setPhishingState(
          confidence,
          "High probability of credential harvesting or phishing activity detected in this URL."
        );
      } else {
        // Green safe aesthetic
        const safeReason = isAllowlisted
          ? "Official verified domain from PhishLens Trusted Allowlist."
          : "Verified legitimate domain structure and safe structural indicators.";
        setSafeState(confidence, safeReason);
      }

      // Render SHAP Risk Factors
      renderRiskFactors(riskFactors, isPhishing, isAllowlisted);

    } catch (err) {
      console.warn("PhishLens API Connection Error:", err);
      setOfflineState(err.message);
    }
  }

  // --------------------------------------------------------------------------
  // 3. Render SHAP Risk Factors List
  // --------------------------------------------------------------------------
  function renderRiskFactors(riskFactors, isPhishing, isAllowlisted) {
    if (!riskFactorsCard || !riskFactorsList) return;

    riskFactorsList.innerHTML = "";

    if (!riskFactors || riskFactors.length === 0) {
      riskFactorsCard.style.display = "none";
      return;
    }

    if (riskCardTitle) {
      if (isPhishing) {
        riskCardTitle.textContent = "Top Phishing Risk Factors (SHAP)";
      } else if (isAllowlisted) {
        riskCardTitle.textContent = "Allowlist Verification";
      } else {
        riskCardTitle.textContent = "Evaluated Attributes (SHAP)";
      }
    }

    riskFactors.forEach((factor) => {
      const li = document.createElement("li");
      li.className = `risk-item ${isPhishing ? "phishing-item" : "safe-item"}`;

      let factorHtml = "";

      // Format SHAP badge if present in the factor string (e.g. "Description (SHAP: +0.23)")
      const shapMatch = factor.match(/\(SHAP:\s*([+-]?[0-9.]+)\)/i);
      if (shapMatch) {
        const textBefore = factor.replace(shapMatch[0], "").trim();
        const shapValue = shapMatch[1];
        factorHtml = `
          <span class="risk-item-bullet"></span>
          <span>${escapeHtml(textBefore)} <span class="shap-badge">SHAP: ${shapValue}</span></span>
        `;
      } else if (factor.toLowerCase().includes("allowlist")) {
        factorHtml = `
          <span class="risk-item-bullet"></span>
          <span>${escapeHtml(factor)} <span class="shap-badge" style="color: var(--color-safe); background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.25);">VERIFIED SAFE</span></span>
        `;
      } else {
        factorHtml = `
          <span class="risk-item-bullet"></span>
          <span>${escapeHtml(factor)}</span>
        `;
      }

      li.innerHTML = factorHtml;
      riskFactorsList.appendChild(li);
    });

    riskFactorsCard.style.display = "flex";
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // --------------------------------------------------------------------------
  // 4. UI State Handlers
  // --------------------------------------------------------------------------
  function setScanningState() {
    statusBlock.className = "status-block scanning";
    statusTitle.textContent = "Scanning...";
    statusDescription.textContent = "Connecting to PhishLens AI engine and calculating SHAP explainability...";
    confidenceMeter.style.display = "none";
    if (riskFactorsCard) riskFactorsCard.style.display = "none";

    // Radar / Searching icon
    iconSvg.innerHTML = `
      <circle cx="12" cy="12" r="10"/>
      <path d="m4.93 4.93 4.24 4.24"/>
      <circle cx="12" cy="12" r="4"/>
    `;
  }

  function setSafeState(confidence, reason) {
    statusBlock.className = "status-block safe";
    statusTitle.textContent = "Legitimate Website";
    statusDescription.textContent = reason;
    confidenceMeter.textContent = `Confidence: ${confidence.toFixed(1)}%`;
    confidenceMeter.style.display = "inline-block";

    // Checkmark Shield icon
    iconSvg.innerHTML = `
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      <path d="m9 12 2 2 4-4"/>
    `;
  }

  function setPhishingState(confidence, reason) {
    statusBlock.className = "status-block phishing";
    statusTitle.textContent = "Phishing Detected!";
    statusDescription.textContent = reason;
    confidenceMeter.textContent = `Risk Score: ${confidence.toFixed(1)}%`;
    confidenceMeter.style.display = "inline-block";

    // Warning Triangle / Alert icon
    iconSvg.innerHTML = `
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    `;
  }

  function setNeutralState(title, desc, badge) {
    statusBlock.className = "status-block";
    statusTitle.textContent = title;
    statusDescription.textContent = desc;
    confidenceMeter.textContent = badge;
    confidenceMeter.style.display = "inline-block";

    // Info icon
    iconSvg.innerHTML = `
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="16" x2="12" y2="12"/>
      <line x1="12" y1="8" x2="12.01" y2="8"/>
    `;
  }

  function setOfflineState(errorMsg) {
    statusBlock.className = "status-block";
    statusTitle.textContent = "Backend Offline";
    statusDescription.textContent = "Cannot reach PhishLens Flask API at http://127.0.0.1:5000. Run 'python app.py' in terminal.";
    confidenceMeter.textContent = "Server Offline";
    confidenceMeter.style.display = "inline-block";

    if (riskFactorsCard) riskFactorsCard.style.display = "none";
    if (backendStatusText) backendStatusText.textContent = "API Offline (python app.py)";

    // Server Disconnected icon
    iconSvg.innerHTML = `
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>
      <line x1="6" y1="6" x2="6.01" y2="6"/>
      <line x1="6" y1="18" x2="6.01" y2="18"/>
    `;
  }

  // --------------------------------------------------------------------------
  // 5. Adaptive Memory - "Report Mistake" Feedback Loop
  // --------------------------------------------------------------------------
  reportMistakeBtn.addEventListener("click", () => {
    if (!activeTabUrl) return;

    const feedbackEntry = {
      url: activeTabUrl,
      timestamp: new Date().toISOString(),
      previousVerdict: currentVerdict ? currentVerdict.prediction : "unknown",
      confidence: currentVerdict ? currentVerdict.confidence_score : null,
      reportedCorrection: currentVerdict && currentVerdict.prediction === "Safe" ? "Phishing" : "Safe"
    };

    // Save report into Chrome storage for Adaptive Memory retrain pipeline
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get({ adaptiveReports: [] }, (data) => {
        const reports = data.adaptiveReports;
        reports.push(feedbackEntry);
        chrome.storage.local.set({ adaptiveReports: reports }, () => {
          showFeedbackSuccess();
        });
      });
    } else {
      console.log("[Adaptive Memory Feedback Stored]:", feedbackEntry);
      showFeedbackSuccess();
    }
  });

  function showFeedbackSuccess() {
    feedbackMessage.style.display = "block";
    reportMistakeBtn.style.opacity = "0.6";
    reportMistakeBtn.disabled = true;

    setTimeout(() => {
      feedbackMessage.style.display = "none";
      reportMistakeBtn.style.opacity = "1";
      reportMistakeBtn.disabled = false;
    }, 3500);
  }
});
