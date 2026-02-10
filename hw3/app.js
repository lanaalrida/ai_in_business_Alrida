// app.js - Sentiment Analysis with GAS Logging (CORS-safe pattern)

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// Constants
const GAS_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbyMxWeRoRh2f1w153NAQKyzeK0uv8wd37nhjJFFYRLLmcqGtzzlTv7A-hAUJIiOpZIs/exec";
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
const apiTokenInput = document.getElementById("api-token"); // For UI compatibility only

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

/**
 * Send one analysis event as a CORS simple request (no preflight).
 * Payload fields are flattened into form data.
 */
async function sendLogSimple(payload) {
  // Prepare form data (CORS-safe, no custom headers)
  const form = new URLSearchParams();
  form.set("ts", String(payload.ts || Date.now()));
  form.set("review", String(payload.review || "").substring(0, 5000));
  form.set("sentiment", String(payload.sentiment || ""));
  form.set("meta", JSON.stringify(payload.meta || {}));

  try {
    const res = await fetch(GAS_WEB_APP_URL, {
      method: "POST",
      body: form // application/x-www-form-urlencoded; no headers to avoid preflight
    });
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }
    
    console.log("Logged to Google Sheets successfully");
    return { success: true };
  } catch (err) {
    console.error("Google Sheets logging failed:", err);
    // Silent fail - don't show error to user since logging is secondary
    return { success: false, error: String(err) };
  }
}

// Initialize the app
document.addEventListener("DOMContentLoaded", function () {
  // Load the TSV file
  loadReviews();

  // Set up event listener for analyze button
  analyzeBtn.addEventListener("click", analyzeRandomReview);

  // Initialize transformers.js sentiment model
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
      if (!response.ok) {
        throw new Error("Failed to load TSV file");
      }
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

  try {
    // Analyze sentiment
    const result = await analyzeSentiment(selectedReview);
    const { sentiment, label, score } = extractSentimentData(result);
    
    // Display result
    displaySentimentResult(sentiment, label, score);
    
    // Log to Google Sheets (fire-and-forget)
    logAnalysis(selectedReview, sentiment, label, score);
    
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
  if (!sentimentPipeline) {
    throw new Error("Sentiment model is not initialized.");
  }

  const output = await sentimentPipeline(text);

  if (!Array.isArray(output) || output.length === 0) {
    throw new Error("Invalid sentiment output from local model.");
  }

  return [output];
}

// Extract sentiment data from result
function extractSentimentData(result) {
  let sentiment = "neutral";
  let score = 0.5;
  let label = "NEUTRAL";

  if (
    Array.isArray(result) &&
    result.length > 0 &&
    Array.isArray(result[0]) &&
    result[0].length > 0
  ) {
    const sentimentData = result[0][0];

    if (sentimentData && typeof sentimentData === "object") {
      label = sentimentData.label ? sentimentData.label.toUpperCase() : "NEUTRAL";
      score = sentimentData.score || 0.5;

      // Determine sentiment bucket
      if (label === "POSITIVE" && score > 0.5) {
        sentiment = "positive";
      } else if (label === "NEGATIVE" && score > 0.5) {
        sentiment = "negative";
      } else {
        sentiment = "neutral";
      }
    }
  }

  return { sentiment, label, score };
}

// Display sentiment result
function displaySentimentResult(sentiment, label, score) {
  sentimentResult.classList.add(sentiment);
  sentimentResult.innerHTML = `
    <i class="fas ${getSentimentIcon(sentiment)} icon"></i>
    <span>${label} (${(score * 100).toFixed(1)}% confidence)</span>
  `;
}

// Log analysis to Google Sheets
async function logAnalysis(review, sentiment, label, score) {
  const userId = getUserId();
  const meta = {
    user_id: userId,
    model: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
    sentiment_bucket: sentiment,
    label: label,
    confidence: score,
    user_agent: navigator.userAgent,
    timestamp_iso: new Date().toISOString(),
    review_length: review.length
  };

  const payload = {
    ts: Date.now(),
    review: review.substring(0, 5000), // Safe truncation
    sentiment: `${label} (${(score * 100).toFixed(1)}% confidence)`,
    meta: meta
  };

  // Send log (fire-and-forget)
  sendLogSimple(payload).catch(err => {
    console.error("Background logging error:", err);
  });
}

// Get appropriate icon for sentiment bucket
function getSentimentIcon(sentiment) {
  switch (sentiment) {
    case "positive":
      return "fa-thumbs-up";
    case "negative":
      return "fa-thumbs-down";
    default:
      return "fa-question-circle";
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
