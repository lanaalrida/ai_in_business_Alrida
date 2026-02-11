// app.js - Sentiment Analysis with Business Logic & GAS Logging
// Preserves your existing structure – only adds:
// - determineBusinessAction(score, label)
// - displayBusinessAction(action)
// - action_taken field in logging

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// Constants
const GAS_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbxe8UyJXOFRTSadcCOvOVjaFMpLKnb9wHLc9QqapiR08clgfWui14EixT_sRthslZxT/exec";
const LS_KEY_UID = "sa_uid";

// Global variables
let reviews = [];
let sentimentPipeline = null;

// DOM elements
const analyzeBtn = document.getElementById("analyze-btn");
const reviewText = document.getElementById("review-text");
const sentimentResult = document.getElementById("sentiment-result");
const loadingElement = document.querySelector(".loading");
const errorElement = document.getElementById("error-message");
const statusElement = document.getElementById("status");
const apiTokenInput = document.getElementById("api-token"); // UI compatibility

// NEW: DOM elements for business action
const actionResult = document.getElementById("action-result");
const actionButtons = document.getElementById("action-buttons");

/** Get or create a stable pseudo user id. */
function getUserId() {
  let uid = localStorage.getItem(LS_KEY_UID);
  if (!uid) {
    uid = (crypto?.randomUUID?.() || Math.random().toString(36).slice(2) + Date.now().toString(36));
    localStorage.setItem(LS_KEY_UID, uid);
  }
  return uid;
}

/** Update status message */
function updateStatus(message, type = "info") {
  if (statusElement) {
    statusElement.textContent = message;
    statusElement.className = `status ${type}`;
  }
}

/* ----------------------------------------------------------------------
   BUSINESS LOGIC – single function as per assignment
---------------------------------------------------------------------- */
/**
 * Determines the appropriate business action based on sentiment analysis.
 * @param {number} confidence - Confidence score (0.0 to 1.0)
 * @param {string} label - Model label (e.g., "POSITIVE", "NEGATIVE")
 * @returns {object} Action metadata: code, message, color, icon, cssClass
 */
function determineBusinessAction(confidence, label) {
  // Normalize to 0 (worst) – 1 (best)
  let normalizedScore;
  if (label === "POSITIVE") {
    normalizedScore = confidence;
  } else if (label === "NEGATIVE") {
    normalizedScore = 1.0 - confidence;
  } else {
    // Fallback for any other label (should not happen with SST-2)
    normalizedScore = 0.5;
  }

  // Apply thresholds
  if (normalizedScore <= 0.4) {
    return {
      actionCode: "OFFER_COUPON",
      uiMessage: "We are truly sorry for your negative experience. Please accept this 50% discount coupon.",
      uiColor: "#dc3545",
      icon: "fa-fire",
      cssClass: "coupon"
    };
  } else if (normalizedScore < 0.7) {
    return {
      actionCode: "REQUEST_FEEDBACK",
      uiMessage: "Thank you for your feedback! Could you tell us how we can improve?",
      uiColor: "#6b7280",
      icon: "fa-clipboard-question",
      cssClass: "feedback"
    };
  } else {
    return {
      actionCode: "ASK_REFERRAL",
      uiMessage: "We're thrilled you enjoyed your experience! Refer a friend and both of you will earn rewards.",
      uiColor: "#3b82f6",
      icon: "fa-user-friends",
      cssClass: "referral"
    };
  }
}

/**
 * Update the UI with the selected business action.
 */
function displayBusinessAction(action) {
  if (!actionResult || !actionButtons) return;

  // Reset and set new class
  actionResult.className = "action-result";
  actionResult.classList.add(action.cssClass);

  // Update icon
  const iconEl = actionResult.querySelector('.action-icon');
  if (iconEl) {
    iconEl.className = `fas ${action.icon} action-icon`;
    iconEl.style.color = action.uiColor;
  }

  // Update message
  const msgEl = actionResult.querySelector('.action-message');
  if (msgEl) {
    msgEl.textContent = action.uiMessage;
    msgEl.style.color = action.uiColor;
  }

  // Update heading
  const headingEl = actionResult.querySelector('h3');
  if (headingEl) {
    headingEl.textContent = `System Decision:`;
    headingEl.style.color = action.uiColor;
  }

  // Generate action buttons
  actionButtons.innerHTML = '';
  if (action.actionCode === "OFFER_COUPON") {
    actionButtons.innerHTML = `
      <button class="action-button coupon-button" onclick="window.generateCoupon()">
        <i class="fas fa-tag"></i> Generate 50% Off Coupon
      </button>
      <button class="action-button" style="background:#718096; color:white;" onclick="window.contactSupport()">
        <i class="fas fa-headset"></i> Contact Support
      </button>
    `;
  } else if (action.actionCode === "REQUEST_FEEDBACK") {
    actionButtons.innerHTML = `
      <a href="https://forms.gle/g1KwfQmetxRGoHQy6" target="_blank" class="action-button feedback-button">
        <i class="fas fa-edit"></i> Complete Survey
      </a>
      <button class="action-button" style="background:#4a5568; color:white;" onclick="window.scheduleCall()">
        <i class="fas fa-phone"></i> Schedule Call
      </button>
    `;
  } else if (action.actionCode === "ASK_REFERRAL") {
    actionButtons.innerHTML = `
      <button class="action-button referral-button" onclick="window.shareReferral()">
        <i class="fas fa-share-alt"></i> Share Referral Link
      </button>
      <button class="action-button" style="background:#2b6cb0; color:white;" onclick="window.writeTestimonial()">
        <i class="fas fa-star"></i> Write Testimonial
      </button>
    `;
  }
}

// ----------------------------------------------------------------------
// Action button handlers (exposed globally for onclick)
window.generateCoupon = function() {
  const code = "SAVE50-" + Math.random().toString(36).substring(2,8).toUpperCase();
  alert(`Your 50% discount coupon: ${code}\nValid for 30 days.`);
};
window.contactSupport = function() {
  alert("Our support team will contact you within 24 hours.");
};
window.scheduleCall = function() {
  alert("Please check your email for scheduling options.");
};
window.shareReferral = function() {
  const link = "https://example.com/ref/" + getUserId().substring(0,8);
  alert(`Share this link with friends: ${link}\nYou both get 20% off!`);
};
window.writeTestimonial = function() {
  window.open("https://example.com/testimonial-form", "_blank");
};
// ----------------------------------------------------------------------

/**
 * Send one analysis event as a CORS simple request (no preflight).
 * Now includes 'action_taken' field.
 */
async function sendLogSimple(payload) {
  const form = new URLSearchParams();
  form.set("ts", String(payload.ts || Date.now()));
  form.set("review", String(payload.review || "").substring(0, 5000));
  form.set("sentiment", String(payload.sentiment || ""));
  form.set("action_taken", String(payload.action_taken || ""));  // NEW COLUMN
  form.set("meta", JSON.stringify(payload.meta || {}));

  try {
    const res = await fetch(GAS_WEB_APP_URL, {
      method: "POST",
      body: form
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    console.log("Logged to Google Sheets successfully");
    return { success: true };
  } catch (err) {
    console.error("Google Sheets logging failed:", err);
    return { success: false, error: String(err) };
  }
}

// Initialize the app
document.addEventListener("DOMContentLoaded", function () {
  loadReviews();
  analyzeBtn.addEventListener("click", analyzeRandomReview);
  initSentimentModel();
});

// Initialize transformers.js text-classification pipeline
async function initSentimentModel() {
  try {
    updateStatus("Loading sentiment model...", "info");
    sentimentPipeline = await pipeline(
      "text-classification",
      "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
    );
    updateStatus("Sentiment model ready", "success");
  } catch (error) {
    console.error("Failed to load sentiment model:", error);
    showError("Failed to load sentiment model. Please check your network connection and try again.");
    updateStatus("Model load failed", "error");
  }
}

// Load and parse the TSV file using Papa Parse
function loadReviews() {
  fetch("reviews_test.tsv")
    .then((response) => {
      if (!response.ok) throw new Error("Failed to load TSV file");
      return response.text();
    })
    .then((tsvData) => {
      Papa.parse(tsvData, {
        header: true,
        delimiter: "\t",
        complete: (results) => {
          reviews = results.data
            .map((row) => row.text)
            .filter((text) => typeof text === "string" && text.trim() !== "");
          console.log("Loaded", reviews.length, "reviews");
          updateStatus(`Loaded ${reviews.length} reviews`, "success");
        },
        error: (error) => {
          console.error("TSV parse error:", error);
          showError("Failed to parse TSV file: " + error.message);
        },
      });
    })
    .catch((error) => {
      console.error("TSV load error:", error);
      showError("Failed to load TSV file: " + error.message);
    });
}

// Analyze a random review
async function analyzeRandomReview() {
  hideError();

  if (!Array.isArray(reviews) || reviews.length === 0) {
    showError("No reviews available. Please try again later.");
    return;
  }

  if (!sentimentPipeline) {
    showError("Sentiment model is not ready yet. Please wait a moment.");
    return;
  }

  const selectedReview = reviews[Math.floor(Math.random() * reviews.length)];

  // Display the review
  reviewText.textContent = selectedReview;

  // Show loading state
  loadingElement.style.display = "block";
  analyzeBtn.disabled = true;
  sentimentResult.innerHTML = "";
  sentimentResult.className = "sentiment-result";

  // Reset action UI
  if (actionResult) {
    actionResult.className = "action-result";
    const iconEl = actionResult.querySelector('.action-icon');
    if (iconEl) iconEl.className = "fas fa-spinner fa-spin action-icon";
    const msgEl = actionResult.querySelector('.action-message');
    if (msgEl) msgEl.textContent = "Analyzing and determining action...";
  }

  try {
    // Analyze sentiment
    const result = await analyzeSentiment(selectedReview);
    const { sentiment, label, score, rawLabel } = extractSentimentData(result); // added rawLabel

    // Display sentiment result
    displaySentimentResult(sentiment, label, score);

    // --- BUSINESS LOGIC INTEGRATION ---
    const action = determineBusinessAction(score, rawLabel);
    displayBusinessAction(action);

    // Log to Google Sheets with action_taken
    logAnalysis(selectedReview, sentiment, label, score, action);
    // ----------------------------------

  } catch (error) {
    console.error("Error:", error);
    showError(error.message || "Failed to analyze sentiment.");
  } finally {
    loadingElement.style.display = "none";
    analyzeBtn.disabled = false;
  }
}

// Analyze sentiment using transformers.js
async function analyzeSentiment(text) {
  if (!sentimentPipeline) throw new Error("Sentiment model is not initialized.");
  const output = await sentimentPipeline(text);
  if (!Array.isArray(output) || output.length === 0) throw new Error("Invalid sentiment output.");
  return [output];
}

// Extract sentiment data – now also returns rawLabel (original model label)
function extractSentimentData(result) {
  let sentiment = "neutral";
  let score = 0.5;
  let label = "NEUTRAL";
  let rawLabel = "NEUTRAL";  // new

  if (Array.isArray(result) && result.length > 0 && Array.isArray(result[0]) && result[0].length > 0) {
    const sentimentData = result[0][0];
    if (sentimentData && typeof sentimentData === "object") {
      rawLabel = sentimentData.label || "NEUTRAL";
      label = rawLabel.toUpperCase();
      score = sentimentData.score || 0.5;

      if (label === "POSITIVE" && score > 0.5) {
        sentiment = "positive";
      } else if (label === "NEGATIVE" && score > 0.5) {
        sentiment = "negative";
      } else {
        sentiment = "neutral";
      }
    }
  }

  return { sentiment, label, score, rawLabel };
}

// Display sentiment result
function displaySentimentResult(sentiment, label, score) {
  sentimentResult.classList.add(sentiment);
  sentimentResult.innerHTML = `
    <i class="fas ${getSentimentIcon(sentiment)} icon"></i>
    <span>${label} (${(score * 100).toFixed(1)}% confidence)</span>
  `;
}

// Log analysis to Google Sheets – now includes action_taken
async function logAnalysis(review, sentiment, label, score, action) {
  const userId = getUserId();
  const meta = {
    user_id: userId,
    model: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
    sentiment_bucket: sentiment,
    label: label,
    confidence: score,
    user_agent: navigator.userAgent,
    timestamp_iso: new Date().toISOString(),
    review_length: review.length,
    business_decision: action.actionCode   // record which action was triggered
  };

  const payload = {
    ts: Date.now(),
    review: review.substring(0, 5000),
    sentiment: `${label} (${(score * 100).toFixed(1)}% confidence)`,
    action_taken: action.actionCode,      // new column
    meta: meta
  };

  sendLogSimple(payload).catch(err => console.error("Background logging error:", err));
}

// Get appropriate icon for sentiment bucket
function getSentimentIcon(sentiment) {
  switch (sentiment) {
    case "positive": return "fa-thumbs-up";
    case "negative": return "fa-thumbs-down";
    default: return "fa-question-circle";
  }
}

// Show error message
function showError(message) {
  errorElement.textContent = message;
  errorElement.style.display = "block";
}

// Hide error message
function hideError() {
  errorElement.style.display = "none";
}
