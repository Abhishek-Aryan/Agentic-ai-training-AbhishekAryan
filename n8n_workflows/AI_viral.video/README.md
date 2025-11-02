🤖 AI Short-Form Video Generation & Auto-Publishing Pipeline

Powered by n8n, OpenRouter, PIAPI, Createmate & Blotato

A fully automated end-to-end system that generates AI battle shorts (like Ape vs Lion) from scratch — complete with scenes, close-ups, aftermath images, video rendering, and multi-platform publishing — all for under $1 per video.

🎬 Overview

This project automates the entire short-form content creation pipeline using AI agents, automation workflows, and low-cost APIs.
The workflow is designed and executed in n8n, and uses Google Sheets for content sourcing, AI models for prompt & image generation, Createmate for rendering, and Blotato for auto-publishing.

🌟 Key Highlights

Automatically generates fight matchups, scenes, and visuals

Uses Flux image model via PIAPI for realistic imagery

Fully template-based video rendering in Createmate

Auto-posts to YouTube Shorts, TikTok, and Instagram

Cost per full video: Under $1

100% reproducible via free workflow JSON and Google Sheet template

🧩 Workflow Architecture
🟩 1. Create Scenes

Generates the main concept and matchups.

Steps:

Triggered from Google Sheet (e.g., “Ape vs Big Cats”).

AI agent (OpenRouter → Chat model) generates 8 matchups like:

Ape vs Lion

Ape vs Tiger

Ape vs Leopard

Ape vs Jaguar
(and so on)

Output: Scene JSON with structured data (main character + opponents).

🟦 2. Create Close-up Images

Generates 16 square close-up images (two per matchup) via Flux/PIAPI.

Steps:

Prompts are generated dynamically from scene text.

Images are generated using PIAPI (POST https://api.piapi.ai/generate).

Waits 90 seconds to ensure completion.

Aggregates all generated images and updates Google Sheets.

Output: 16 intimidating close-up images ready for video.

🟪 3. Create Winner Images

Determines winners and generates portrait-style aftermath visuals.

Steps:

Second AI agent (OpenRouter → Chat model) evaluates likely winners.

Generates 8 winner prompts (e.g., “lion stands victorious over ape”).

PIAPI generates the final set of portrait aftermath images.

Data is aggregated and synced with Sheets.

Output: 8 realistic winner/aftermath images.

🟥 4. Render Video

All generated images are compiled into a short cinematic video using Createmate.

Steps:

Fetches all image URLs from Google Sheets.

Uses a predefined Createmate template for layout and transitions.

Sends render job via API (POST https://api.createmate.ai/render).

Waits 90 seconds for video generation.

Saves final video link to Google Sheets.

Output: 1 complete, high-quality short video.

🟧 5. Auto-Publish to Platforms

Uploads the generated video to YouTube Shorts, TikTok, and Instagram automatically via Blotato.

Steps:

Blotato API handles uploads and scheduling.

Video titles, descriptions, and tags are dynamically pulled from Google Sheets.

Output: Multi-platform published video.

⚙️ Setup Guide

Make a copy of this Google Sheet Template

Connect it to all 5 Google Sheet nodes inside n8n.

Connect your OpenRouter API key

Used in the two Chat Model nodes for scene and winner generation.

Create a PIAPI
 account

Get your API key and add it to the image generation nodes.

Create a Createmate
 account

Connect your Template ID and Account ID.

Duplicate the same video template shown in the reference video using the same Skoop point.

Connect your Blotato
 account

Add your API key to enable auto-publishing to Instagram, TikTok, and YouTube.

💡 Use promo code NATE30 for 30% off first 6 months.

💰 Cost Breakdown per Video
Service	Purpose	Approx Cost
OpenRouter (AI Agents)	Scene & prompt generation	$0.01
PIAPI (Flux Image Model)	24 total images (16 close-ups + 8 winners)	$0.36
Createmate	Video rendering (Essential Plan)	$0.35
Blotato + Sheets	Publishing & data sync	~$0.20
💵 Total	—	~$0.92 per video
🧠 Tools & Tech Stack
Component	Tool	Purpose
Automation	n8n	Workflow orchestration
Prompt & Story AI	OpenRouter (Gemini, Mistral, Claude etc.)	Scene & text generation
Image Generation	PIAPI (Flux model)	Visual generation
Rendering Engine	Createmate	Video compilation
Data Storage	Google Sheets	Content, metadata, URLs
Publishing API	Blotato	Multi-platform posting

🚀 Quick Start

Import the Workflow

In n8n → “Import Workflow” → Upload ai_video_automation.json

Connect Your Credentials

Google Sheets, OpenRouter, PIAPI, Createmate, Blotato

Customize Template

Update your own Createmate template and test rendering.

Run the Workflow

Trigger manually or schedule it with n8n’s built-in Scheduler node.

Check Results

Generated assets and video URLs appear in Google Sheets.

📊 Example Output

Prompt Example: “Ape vs Lion, intense jungle battle at dawn.”

Generated Assets:

16 Close-up Images

8 Winner Portraits

1 Rendered Video

Auto Uploaded To:

✅ YouTube Shorts

✅ TikTok

✅ Instagram

🧩 Future Improvements

🔊 Add AI voiceovers using ElevenLabs or OpenAI TTS

🈵 Multi-language prompt generation

✍️ Dynamic captions & subtitles via Whisper or Gemini

📅 Scheduled content calendars synced to Sheets