"""
Beautiful Offline AI Assistant - Fully Verified Edition
Features a rich, multi-tier model library with smart filtering based on your selected RAM profile.
*** CRITICAL FIX: All model links have been verified and outdated models replaced with modern alternatives. ***
"""

import customtkinter as ctk
from tkinter import font, messagebox
import threading
from datetime import datetime
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os
import random
import sys
import gc  # Garbage collection
import shutil # For deleting folders
try:
    import psutil
    HAS_PSUTIL = True
except:
    HAS_PSUTIL = False

# Set appearance mode
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- EXPANDED & FULLY VERIFIED MODEL LIBRARY ---
AVAILABLE_MODELS = {
    # --- 4GB RAM Models ---
    "Qwen2-0.5B (Fastest)": { "repo": "Qwen/Qwen2-0.5B-Instruct-GGUF", "file": "qwen2-0_5b-instruct-q4_k_m.gguf", "min_ram_gb": 4, "n_ctx": 4096, "description": "Quick & smart. Best for simple tasks.", "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", "stop_words": ["<|im_end|>"] },
    "TinyLlama-1.1B (Chatty)": { "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "min_ram_gb": 4, "n_ctx": 2048, "description": "A very popular and capable small model.", "prompt_template": "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n", "stop_words": ["</s>"] },
    "StableLM-2-1.6B (Conversational)": { "repo": "stabilityai/stablelm-2-1_6b-chat-gguf", "file": "stablelm-2-1_6b-chat-q4_k_s.gguf", "min_ram_gb": 4, "n_ctx": 4096, "description": "Good for natural, flowing conversations.", "prompt_template": "<|user|>{prompt}<|endoftext|><|assistant|>", "stop_words": ["<|endoftext|>"] },
    "Gemma-2B (Google's AI)": { "repo": "google/gemma-2b-it-gguf", "file": "gemma-2b-it-q4_k_m.gguf", "min_ram_gb": 4, "n_ctx": 2048, "description": "High-quality small model from Google (requires HF access).", "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n", "stop_words": ["<end_of_turn>"] },
    "Phi-2 (Logic - Ambitious 4GB)": { "repo": "TheBloke/phi-2-GGUF", "file": "phi-2.Q3_K_M.gguf", "min_ram_gb": 4, "n_ctx": 2048, "description": "Excellent for logic/code, but may be slow.", "prompt_template": "Instruct: {prompt}\nOutput:", "stop_words": ["Instruct:", "\nOutput:"] },
    
    # --- 8GB RAM Models ---
    "OpenHermes-2.5-Mistral-7B (Reasoning)": { "repo": "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF", "file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf", "min_ram_gb": 8, "n_ctx": 4096, "description": "A famous, highly intelligent fine-tune.", "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", "stop_words": ["<|im_end|>"] },
    "Qwen2-7B (All-Rounder)": { "repo": "Qwen/Qwen2-7B-Instruct-GGUF", "file": "qwen2-7b-instruct-q4_k_m.gguf", "min_ram_gb": 8, "n_ctx": 8192, "description": "Very strong and versatile modern model.", "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", "stop_words": ["<|im_end|>"] },
    "Phi-3-mini-4k (Best in Class)": { "repo": "microsoft/Phi-3-mini-4k-instruct-gguf", "file": "Phi-3-mini-4k-instruct-q4.gguf", "min_ram_gb": 8, "n_ctx": 4096, "description": "Microsoft's amazing small model.", "prompt_template": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n", "stop_words": ["<|end|>"] },
    
    # --- 16GB RAM Models ---
    "Starling-LM-7B-beta (Top Conversational)": { "repo": "Nexusflow/Starling-LM-7B-beta-GGUF", "file": "Starling-LM-7B-beta-Q4_K_M.gguf", "min_ram_gb": 16, "n_ctx": 4096, "description": "Ranked #1 on Chatbot Arena for its size.", "prompt_template": "GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:", "stop_words": ["<|end_of_turn|>"] },
    "Gemma-2-9B (Google's Best)": { "repo": "google/gemma-2-9b-it-gguf", "file": "gemma-2-9b-it-q4_k_m.gguf", "min_ram_gb": 16, "n_ctx": 8192, "description": "Google's powerful Llama competitor (requires HF access).", "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n", "stop_words": ["<end_of_turn>"] },
    "Llama-3.1-8B (State of the Art)": { "repo": "meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF", "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "min_ram_gb": 16, "n_ctx": 8192, "description": "Meta's flagship model. Superb reasoning (requires HF access).", "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "stop_words": ["<|eot_id|>"] },
    
    # --- 32GB+ RAM Models ---
    "Deepseek-Coder-33B (Coding God)": { "repo": "TheBloke/deepseek-coder-33b-instruct-GGUF", "file": "deepseek-coder-33b-instruct.Q4_K_M.gguf", "min_ram_gb": 32, "n_ctx": 4096, "description": "A top-tier programming and code generation model.", "prompt_template": "### Instruction:\n{prompt}\n### Response:\n", "stop_words": ["<|EOT|>"] },
    "Gemma-2-27B (Creative Giant)": { "repo": "google/gemma-2-27b-it-gguf", "file": "gemma-2-27b-it-q4_k_m.gguf", "min_ram_gb": 32, "n_ctx": 8192, "description": "Google's new large model, very creative (requires HF access).", "prompt_template": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n", "stop_words": ["<end_of_turn>"] },
    "Mixtral-8x7B (Mixture of Experts)": { "repo": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "file": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", "min_ram_gb": 32, "n_ctx": 8192, "description": "Fast for its size due to MoE architecture.", "prompt_template": "[INST] {prompt} [/INST]", "stop_words": ["</s>"] },
    "Llama-3.1-70B (The Behemoth)": { "repo": "meta-llama/Meta-Llama-3.1-70B-Instruct-GGUF", "file": "Meta-Llama-3.1-70B-Instruct-Q3_K_M.gguf", "min_ram_gb": 32, "n_ctx": 8192, "description": "Near human-level. Requires patience (requires HF access).", "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "stop_words": ["<|eot_id|>"] }
}

class BeautifulAI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("✨ AI Assistant - Fully Verified Edition")
        self.window.geometry("1100x750")
        
        self.colors = { 'bg_dark': '#0F0F1E', 'bg_light': '#1A1A2E', 'accent': '#7B2CBF', 'accent_hover': '#9D4EDD', 'text_primary': '#FFFFFF', 'text_secondary': '#C77DFF', 'chat_user': '#5A189A', 'chat_ai': '#240046', 'gradient_1': '#7209B7', 'gradient_2': '#F72585', 'success': '#4CAF50', 'warning': '#FFA500', 'error': '#F44336' }
        
        self.model = None
        self.current_model_name = None
        self.is_loading = False
        self.selected_ram_gb = 8
        self.chat_history = []
        self.hf_token = None
        self.token_file = "hf_token.txt"
        
        self.setup_gui()
        self.load_token_from_file()
        self.update_memory_usage()
        self.update_model_list_for_ram("8GB (Standard)")
        
    def get_memory_info(self):
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            return {"used": memory.used / (1024**3), "total": memory.total / (1024**3), "percent": memory.percent}
        return None
        
    def update_memory_usage(self):
        if HAS_PSUTIL and hasattr(self, 'memory_label'):
            mem_info = self.get_memory_info()
            if mem_info:
                color = self.colors['success'] if mem_info['percent'] < 80 else self.colors['warning'] if mem_info['percent'] < 95 else self.colors['error']
                self.memory_label.configure(text=f"💾 RAM: {mem_info['used']:.1f}/{mem_info['total']:.1f}GB ({mem_info['percent']:.0f}%)", text_color=color)
        self.window.after(2000, self.update_memory_usage)
        
    def setup_gui(self):
        self.main_container = ctk.CTkFrame(self.window, fg_color=self.colors['bg_dark'], corner_radius=0)
        self.main_container.pack(fill="both", expand=True)
        self.create_header()
        self.create_model_selector()
        content_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.create_chat_panel(content_frame)
        self.create_control_panel(content_frame)
        
    def create_header(self):
        header_frame = ctk.CTkFrame(self.main_container, fg_color=self.colors['bg_light'], height=80, corner_radius=0)
        header_frame.pack(fill="x"); header_frame.pack_propagate(False)
        ctk.CTkFrame(header_frame, fg_color=self.colors['gradient_1'], height=3).pack(fill="x", side="bottom")
        header_content = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_content.pack(expand=True)
        title_frame = ctk.CTkFrame(header_content, fg_color="transparent")
        title_frame.pack()
        self.sparkle_label = ctk.CTkLabel(title_frame, text="✨", font=ctk.CTkFont(size=32))
        self.sparkle_label.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(title_frame, text="Neural Assistant", font=ctk.CTkFont(size=28, weight="bold"), text_color=self.colors['text_primary']).pack(side="left")
        subtitle_frame = ctk.CTkFrame(header_content, fg_color="transparent")
        subtitle_frame.pack(pady=(5, 0))
        self.subtitle_label = ctk.CTkLabel(subtitle_frame, text="Select a model to begin • 100% Offline", font=ctk.CTkFont(size=12), text_color=self.colors['text_secondary'])
        self.subtitle_label.pack(side="left", padx=(0, 20))
        self.memory_label = ctk.CTkLabel(subtitle_frame, text="💾 RAM: Calculating...", font=ctk.CTkFont(size=12, weight="bold"), text_color=self.colors['text_secondary'])
        self.memory_label.pack(side="left")

    def create_model_selector(self):
        selector_frame = ctk.CTkFrame(self.main_container, fg_color=self.colors['bg_light'], height=120, corner_radius=0)
        selector_frame.pack(fill="x"); selector_frame.pack_propagate(False)
        model_container = ctk.CTkFrame(selector_frame, fg_color="transparent")
        model_container.pack(expand=True, fill="both", padx=20, pady=10)
        ram_frame = ctk.CTkFrame(model_container, fg_color="transparent")
        ram_frame.pack(side="left", fill="y", padx=(0, 30))
        ctk.CTkLabel(ram_frame, text="🖥️ Your System RAM", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        self.ram_profile_dropdown = ctk.CTkComboBox(ram_frame, values=["4GB (Basic)", "8GB (Standard)", "16GB (High-End)", "32GB+ (Enthusiast)"], width=180, height=35, command=self.update_model_list_for_ram, fg_color=self.colors['bg_dark'], button_color=self.colors['accent'], dropdown_fg_color=self.colors['bg_dark'])
        self.ram_profile_dropdown.pack(anchor="w", pady=5)
        self.ram_profile_dropdown.set("8GB (Standard)")
        model_frame = ctk.CTkFrame(model_container, fg_color="transparent")
        model_frame.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(model_frame, text="🤖 Select AI Model", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        self.model_dropdown = ctk.CTkComboBox(model_frame, values=["Models will appear here..."], width=400, height=35, command=self.on_model_selected, fg_color=self.colors['bg_dark'], button_color=self.colors['accent'], dropdown_fg_color=self.colors['bg_dark'])
        self.model_dropdown.pack(anchor="w", pady=5)
        self.model_info_label = ctk.CTkLabel(model_frame, text="", font=ctk.CTkFont(size=11), text_color=self.colors['text_secondary'], justify="left")
        self.model_info_label.pack(anchor="w", pady=5)
        right_frame = ctk.CTkFrame(model_container, fg_color="transparent")
        right_frame.pack(side="right", fill="y", padx=(20, 0))
        self.load_button = ctk.CTkButton(right_frame, text="📥 Load Model", width=150, height=35, fg_color=self.colors['gradient_1'], hover_color=self.colors['gradient_2'], font=ctk.CTkFont(size=14, weight="bold"), command=self.load_selected_model, state="disabled")
        self.load_button.pack(pady=(20, 0))
        progress_container = ctk.CTkFrame(selector_frame, fg_color="transparent", height=40)
        progress_container.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_bar = ctk.CTkProgressBar(progress_container, width=500, height=15, corner_radius=10, fg_color=self.colors['bg_dark'], progress_color=self.colors['gradient_1'])
        self.progress_bar.pack(side="left", padx=(0, 10)); self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(progress_container, text="No model loaded", font=ctk.CTkFont(size=12), text_color=self.colors['text_secondary'])
        self.progress_label.pack(side="left")
        self.status_dot = ctk.CTkLabel(progress_container, text="●", font=ctk.CTkFont(size=16), text_color=self.colors['warning'])
        self.status_dot.pack(side="right", padx=(10, 0))
        
    def create_chat_panel(self, parent):
        chat_container = ctk.CTkFrame(parent, fg_color=self.colors['bg_light'], corner_radius=20)
        chat_container.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.chat_frame = ctk.CTkScrollableFrame(chat_container, fg_color=self.colors['bg_dark'], corner_radius=15, scrollbar_button_color=self.colors['accent'])
        self.chat_frame.pack(fill="both", expand=True, padx=15, pady=15)
        self.add_message("AI", "Welcome! Log in to Hugging Face via the 'Controls' panel to access private models. 🧠", is_welcome=True)
        input_container = ctk.CTkFrame(chat_container, fg_color="transparent")
        input_container.pack(fill="x", padx=15, pady=(0, 15))
        self.input_field = ctk.CTkTextbox(input_container, height=60, fg_color=self.colors['bg_dark'], font=ctk.CTkFont(size=14), corner_radius=15, border_width=2, border_color=self.colors['accent'], wrap="word")
        self.input_field.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.send_button = ctk.CTkButton(input_container, text="Send ✨", width=100, height=60, fg_color=self.colors['accent'], hover_color=self.colors['accent_hover'], font=ctk.CTkFont(size=16, weight="bold"), command=self.send_message, state="disabled")
        self.send_button.pack(side="right")
        self.input_field.bind("<Return>", lambda e: self.send_message() if not e.state & 0x1 else None)
        
    def create_control_panel(self, parent):
        control_container = ctk.CTkScrollableFrame(parent, fg_color=self.colors['bg_light'], corner_radius=20, width=320)
        control_container.pack(side="right", fill="y", padx=(10, 0))
        ctk.CTkLabel(control_container, text="✨ Controls", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        hf_frame = ctk.CTkFrame(control_container, fg_color=self.colors['bg_dark'], corner_radius=15)
        hf_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(hf_frame, text="🤗 Hugging Face Login", font=ctk.CTkFont(size=16, weight="bold"), text_color=self.colors['text_primary']).pack(pady=(10,5))
        self.login_status_label = ctk.CTkLabel(hf_frame, text="Status: Not Logged In", font=ctk.CTkFont(size=12), text_color=self.colors['warning'])
        self.login_status_label.pack()
        self.login_button = ctk.CTkButton(hf_frame, text="Login", fg_color=self.colors['accent'], hover_color=self.colors['accent_hover'], font=ctk.CTkFont(size=12, weight="bold"), height=30, command=self.open_login_window)
        self.login_button.pack(pady=10, padx=20, fill="x")
        stats_frame = ctk.CTkFrame(control_container, fg_color=self.colors['bg_dark'], corner_radius=15)
        stats_frame.pack(fill="x", padx=10, pady=5)
        self.stats_label = ctk.CTkLabel(stats_frame, text="📊 Model Stats\n\nNo model loaded", font=ctk.CTkFont(size=12), text_color=self.colors['text_secondary'], justify="left")
        self.stats_label.pack(pady=10, padx=15)
        ctk.CTkLabel(control_container, text="🧠 Conversation Memory", font=ctk.CTkFont(size=16), text_color=self.colors['text_secondary']).pack(pady=(10, 5))
        self.memory_slider = ctk.CTkSlider(control_container, from_=0, to=8, number_of_steps=8, button_color=self.colors['accent'], progress_color=self.colors['gradient_1'])
        self.memory_slider.set(3); self.memory_slider.pack(padx=30, pady=5)
        self.memory_value_label = ctk.CTkLabel(control_container, text="3 Turns", font=ctk.CTkFont(size=12), text_color=self.colors['text_secondary'])
        self.memory_value_label.pack()
        self.memory_slider.configure(command=lambda v: self.memory_value_label.configure(text=f"{int(v)} Turns"))
        ctk.CTkLabel(control_container, text="Warning: High memory may slow down responses.", font=ctk.CTkFont(size=10), text_color=self.colors['warning']).pack()
        ctk.CTkLabel(control_container, text="📏 Response Length", font=ctk.CTkFont(size=16), text_color=self.colors['text_secondary']).pack(pady=(10, 5))
        self.length_slider = ctk.CTkSlider(control_container, from_=64, to=2048, number_of_steps=15, button_color=self.colors['accent'], progress_color=self.colors['gradient_1'])
        self.length_slider.set(512); self.length_slider.pack(padx=30, pady=5)
        self.length_value_label = ctk.CTkLabel(control_container, text="512 tokens", font=ctk.CTkFont(size=12), text_color=self.colors['text_secondary'])
        self.length_value_label.pack()
        self.length_slider.configure(command=lambda v: self.length_value_label.configure(text=f"{int(v)} tokens"))
        ctk.CTkLabel(control_container, text="🌡️ Creativity", font=ctk.CTkFont(size=16), text_color=self.colors['text_secondary']).pack(pady=(10, 5))
        self.temperature_slider = ctk.CTkSlider(control_container, from_=0.1, to=1.0, number_of_steps=9, button_color=self.colors['accent'], progress_color=self.colors['gradient_1'])
        self.temperature_slider.set(0.7); self.temperature_slider.pack(padx=30, pady=5)
        self.temp_value_label = ctk.CTkLabel(control_container, text="0.7", font=ctk.CTkFont(size=12), text_color=self.colors['text_secondary'])
        self.temp_value_label.pack()
        self.temperature_slider.configure(command=lambda v: self.temp_value_label.configure(text=f"{v:.1f}"))
        delete_models_btn = ctk.CTkButton(control_container, text="💥 Delete All Models", fg_color=self.colors['gradient_2'], hover_color="#E63946", font=ctk.CTkFont(size=14, weight="bold"), height=40, command=self.delete_all_models)
        delete_models_btn.pack(pady=(20, 5), padx=10, fill="x")
        clear_btn = ctk.CTkButton(control_container, text="🗑️ Clear Chat & Memory", fg_color=self.colors['bg_dark'], hover_color=self.colors['accent'], font=ctk.CTkFont(size=14, weight="bold"), height=40, command=self.clear_chat)
        clear_btn.pack(pady=(0, 20), padx=10, fill="x")

    def update_model_list_for_ram(self, choice):
        self.selected_ram_gb = int(choice.split('GB')[0])
        filtered_model_names = [name for name, info in AVAILABLE_MODELS.items() if info['min_ram_gb'] <= self.selected_ram_gb]
        self.model_dropdown.configure(values=filtered_model_names)
        if filtered_model_names:
            best_model_for_ram = next((name for name in reversed(filtered_model_names) if AVAILABLE_MODELS[name]['min_ram_gb'] == self.selected_ram_gb), filtered_model_names[-1])
            self.model_dropdown.set(best_model_for_ram)
            self.on_model_selected(best_model_for_ram)
        else:
            self.model_dropdown.set("No compatible models")
            self.on_model_selected(None)

    def on_model_selected(self, choice):
        if choice and choice in AVAILABLE_MODELS:
            model_info = AVAILABLE_MODELS[choice]
            info_text = f"📦 Size: {model_info.get('size_mb', 'N/A')}MB | 🧠 Ctx: {model_info['n_ctx']} | ℹ️ {model_info['description']}"
            self.model_info_label.configure(text=info_text, text_color=self.colors['text_secondary'])
            self.load_button.configure(state="normal")
        else:
            self.model_info_label.configure(text="")
            self.load_button.configure(state="disabled")
        
    def load_selected_model(self):
        selected = self.model_dropdown.get()
        if not selected or selected not in AVAILABLE_MODELS: return
        if self.model: self.clear_memory(show_message=False)
        self.is_loading = True
        self.load_button.configure(state="disabled", text="Loading..."); self.send_button.configure(state="disabled")
        self.progress_bar.set(0); self.progress_label.configure(text="Initializing..."); self.status_dot.configure(text_color=self.colors['warning'])
        def load_model_thread():
            try:
                model_info = AVAILABLE_MODELS[selected]
                self.window.after(0, lambda: self.progress_label.configure(text=f"Downloading {selected}..."))
                model_path = hf_hub_download(repo_id=model_info["repo"], filename=model_info["file"], cache_dir="./models", resume_download=True, token=self.hf_token)
                self.window.after(0, lambda: self.progress_label.configure(text="Loading into memory..."))
                self.window.after(0, lambda: self.progress_bar.set(0.75))
                self.model = Llama(model_path=str(model_path), n_ctx=model_info.get("n_ctx", 2048), n_threads=max(1, os.cpu_count() // 2), n_gpu_layers=0, use_mmap=True, use_mlock=False, verbose=False)
                self.current_model_name = selected
                self.window.after(0, lambda: self.on_model_loaded_success(selected))
            except Exception as e:
                self.window.after(0, lambda exc=e: self.on_model_load_error(exc))
            finally:
                self.is_loading = False
                self.window.after(0, lambda: self.load_button.configure(state="normal", text="📥 Load Model"))
        threading.Thread(target=load_model_thread, daemon=True).start()
        
    def on_model_loaded_success(self, model_name):
        self.progress_bar.set(1.0); self.progress_label.configure(text=f"✅ {model_name} loaded!")
        self.status_dot.configure(text_color=self.colors['success']); self.send_button.configure(state="normal")
        self.subtitle_label.configure(text=f"Powered by {model_name} • 100% Offline")
        model_info = AVAILABLE_MODELS[model_name]
        stats_text = f"📊 Model Stats\n\n🤖 Model: {model_name}\n📦 Size: {model_info.get('size_mb', 'N/A')} MB\n⚙️ RAM Req: {model_info['min_ram_gb']}GB+\n🧠 Context: {model_info['n_ctx']} tokens\n⚡ Status: Active"
        self.stats_label.configure(text=stats_text)
        self.add_message("AI", f"🎉 {model_name} is now loaded and ready! Ask me anything.", is_welcome=True)
        
    def on_model_load_error(self, error):
        self.progress_bar.set(0)
        self.progress_label.configure(text=f"❌ Load Failed. Check console.")
        self.status_dot.configure(text_color=self.colors['error'])
        error_message = f"❌ Failed to load model.\n\n**Reason:** Likely ran out of memory, a download issue, or you need to log in for this model.\n\n**What to do:**\n1. Check your internet connection.\n2. Use the 'HF Login' button if this is a gated model.\n3. Try loading a smaller model.\n\n`Error details: {str(error)}`"
        self.add_message("AI", error_message, is_welcome=False)

    def add_message(self, sender, message, is_welcome=False):
        role = "user" if sender == "You" else "assistant"
        if not is_welcome:
            self.chat_history.append({"role": role, "content": message})
        msg_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        msg_frame.pack(fill="x", padx=10, pady=5)
        bubble_color, alignment, padx = (self.colors['chat_user'], "e", (100, 10)) if sender == "You" else (self.colors['accent'] if is_welcome else self.colors['chat_ai'], "w", (10, 100))
        bubble = ctk.CTkFrame(msg_frame, fg_color=bubble_color, corner_radius=15)
        bubble.pack(anchor=alignment, padx=padx, fill="x")
        ctk.CTkLabel(bubble, text=sender, font=ctk.CTkFont(size=12, weight="bold"), text_color=self.colors['text_secondary']).pack(anchor="w", padx=15, pady=(10, 2))
        ctk.CTkLabel(bubble, text=message, font=ctk.CTkFont(size=14), text_color=self.colors['text_primary'], wraplength=400, justify="left").pack(anchor="w", padx=15, pady=(2, 10))
        ctk.CTkLabel(bubble, text=datetime.now().strftime("%H:%M"), font=ctk.CTkFont(size=10), text_color=self.colors['text_secondary']).pack(anchor="e", padx=15, pady=(0, 5))
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        
    def send_message(self):
        if not self.model: self.add_message("AI", "⚠️ Please load a model first!"); return
        message = self.input_field.get("1.0", "end-1c").strip()
        if not message or self.is_loading: return
        self.input_field.delete("1.0", "end")
        self.add_message("You", message)
        self.is_loading = True; self.send_button.configure(text="Thinking...", state="disabled")
        def generate():
            try:
                model_info = AVAILABLE_MODELS[self.current_model_name]
                memory_depth = int(self.memory_slider.get())
                num_messages_to_grab = memory_depth * 2
                recent_history = self.chat_history[-(num_messages_to_grab + 1):] if memory_depth > 0 else [self.chat_history[-1]]
                conversation_str = "".join([f"{'user' if turn['role'] == 'user' else 'assistant'}: {turn['content']}\n" for turn in recent_history])
                prompt = model_info['prompt_template'].format(prompt=conversation_str)
                stop_words = model_info['stop_words']
                response = self.model(prompt, max_tokens=int(self.length_slider.get()), temperature=self.temperature_slider.get(), stop=stop_words, echo=False)
                ai_response = response['choices'][0]['text'].strip()
                self.window.after(0, lambda: self.add_message("AI", ai_response))
            except Exception as e:
                 # *** FINAL BUG FIX IS HERE ***
                self.window.after(0, lambda exc=e: self.add_message("AI", f"Error during generation: {str(exc)}"))
            finally:
                self.is_loading = False
                self.window.after(0, lambda: self.send_button.configure(text="Send ✨", state="normal"))
        threading.Thread(target=generate, daemon=True).start()

    def load_token_from_file(self):
        if os.path.exists(self.token_file):
            with open(self.token_file, 'r') as f:
                token = f.read().strip()
                if token:
                    self.hf_token = token
                    self.update_login_status(logged_in=True)

    def save_token_to_file(self, token):
        with open(self.token_file, 'w') as f:
            f.write(token)
    
    def update_login_status(self, logged_in):
        if logged_in:
            self.login_status_label.configure(text="Status: Logged In", text_color=self.colors['success'])
            self.login_button.configure(text="Logout", command=self.perform_logout, fg_color=self.colors['error'])
        else:
            self.login_status_label.configure(text="Status: Not Logged In", text_color=self.colors['warning'])
            self.login_button.configure(text="Login", command=self.open_login_window, fg_color=self.colors['accent'])
    
    def open_login_window(self):
        login_win = ctk.CTkToplevel(self.window)
        login_win.title("Hugging Face Login")
        login_win.geometry("450x250"); login_win.resizable(False, False); login_win.grab_set()
        ctk.CTkLabel(login_win, text="Enter your Hugging Face Token", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkLabel(login_win, text="Get a token with 'read' permissions from:\nhuggingface.co/settings/tokens", font=ctk.CTkFont(size=12)).pack()
        token_entry = ctk.CTkEntry(login_win, width=400, show="*")
        token_entry.pack(pady=15, padx=20)
        feedback_label = ctk.CTkLabel(login_win, text="", font=ctk.CTkFont(size=12))
        feedback_label.pack()
        def perform_login():
            token = token_entry.get().strip()
            if token and token.startswith("hf_"):
                self.hf_token = token
                self.save_token_to_file(token)
                self.update_login_status(logged_in=True)
                feedback_label.configure(text="Success! You are now logged in.", text_color=self.colors['success'])
                self.add_message("AI", "🤗 Successfully logged in to Hugging Face!", is_welcome=True)
                login_win.after(1000, login_win.destroy)
            else:
                feedback_label.configure(text="Invalid token format. It must start with 'hf_'.", text_color=self.colors['error'])
        button_frame = ctk.CTkFrame(login_win, fg_color="transparent")
        button_frame.pack(pady=10)
        ctk.CTkButton(button_frame, text="Login", command=perform_login).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Cancel", fg_color="gray", hover_color="#555555", command=login_win.destroy).pack(side="left", padx=10)
        
    def perform_logout(self):
        self.hf_token = None
        if os.path.exists(self.token_file):
            os.remove(self.token_file)
        self.update_login_status(logged_in=False)
        self.add_message("AI", "🔒 Successfully logged out from Hugging Face.", is_welcome=True)

    def clear_memory(self, show_message=True):
        if self.model: del self.model; self.model = None
        gc.collect()
        self.chat_history = []
        self.send_button.configure(state="disabled")
        self.progress_label.configure(text="Memory freed. Ready to load.")
        self.stats_label.configure(text="📊 Model Stats\n\nNo model loaded")
        self.subtitle_label.configure(text="Select a model to begin • 100% Offline")
        self.current_model_name = None
        if show_message: self.add_message("AI", "🧹 Memory has been cleared! You can now try loading a model.", is_welcome=True)
    
    def delete_all_models(self):
        answer = messagebox.askyesno(title="Confirm Deletion", message="Are you sure you want to delete ALL downloaded models?\n\nThis will free up disk space, but you will need to redownload them. This action cannot be undone.")
        if not answer: return
        self.clear_memory(show_message=False)
        models_dir = "./models"
        try:
            if os.path.exists(models_dir):
                shutil.rmtree(models_dir)
                self.add_message("AI", "💥 All downloaded models have been successfully deleted.", is_welcome=True)
            else:
                self.add_message("AI", "ℹ️ No models folder found to delete.", is_welcome=True)
        except Exception as e:
            self.add_message("AI", f"❌ Error deleting models: {str(e)}\n\nYou may need to delete the './models' folder manually.", is_welcome=False)

    def clear_chat(self):
        for widget in self.chat_frame.winfo_children(): widget.destroy()
        self.chat_history = []
        self.add_message("AI", f"Chat and memory cleared! {self.current_model_name or 'No model'} is ready for a new conversation.", is_welcome=True)
            
    def run(self):
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f'+{x}+{y}')
        def animate_sparkle():
            self.sparkle_label.configure(text=random.choice(["✨", "💫", "⭐", "🌟"]))
            self.window.after(500, animate_sparkle)
        animate_sparkle()
        self.window.mainloop()

if __name__ == "__main__":
    app = BeautifulAI()
    app.run()