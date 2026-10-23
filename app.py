                    st.subheader("Audio track analysis")
                    with st.spinner("Extracting video audio and running audio model..."):
                        # Use a unique temp path without pre-creating the file.
                        fd, audio_path = tempfile.mkstemp(suffix=".wav")
                        os.close(fd)
                        os.remove(audio_path)
                        extract_audio_from_video(video_path, audio_path)
                        audio_result = predict_audio_file(audio_path)

                    st.success("Audio track extracted.")
                    show_result(audio_result)
                    st.write(f"Audio duration: {audio_result['duration_seconds']} seconds")
                    st.write(f"Model sample rate: {audio_result['sample_rate']} Hz")

                    with open(audio_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/wav")

                    report["audio_analysis"] = audio_result

                if report["visual_analysis"] is not None or report["audio_analysis"] is not None:
                    st.download_button(
                        "Download combined video forensic report (JSON)",
                        json.dumps(report, indent=2),
                        file_name="deeptrace_video_forensic_report.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Video analysis failed: {e}")
                st.info(
                    "If extraction fails, the video may have no audio track. "
                    "If model loading fails, check HF_TOKEN, the Hugging Face repository, "
                    "and the audio dependencies."
                )
            finally:
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
                for path in (video_path, audio_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass


# ---------------- STANDALONE AUDIO DETECTION ----------------
else:
    st.subheader("🎙️ Audio deepfake detection")
    st.write(
        "Upload a WAV, FLAC, or MP3 file here. To analyze audio from a video, "
        "choose Video detection instead."
    )

    uploaded = st.file_uploader(
        "Upload an audio file", type=["wav", "flac", "mp3"], key="audio_upload"
    )

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes)
        st.write(f"File: {uploaded.name}")
        st.write(f"Size: {len(audio_bytes) / 1024:.1f} KB")
        st.write(f"SHA-256: `{file_hash(audio_bytes)}`")

        if st.button("Analyze audio", type="primary"):
            audio_path = None
            try:
                suffix = os.path.splitext(uploaded.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    audio_path = temp_file.name

                with st.spinner("Loading audio model and analyzing audio..."):
                    result = predict_audio_file(audio_path)

                show_result(result)
                report = {
                    "file_name": uploaded.name,
                    "file_sha256": file_hash(audio_bytes),
                    "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
                    "model_repo": AUDIO_REPO_ID,
                    "result": result,
                    "disclaimer": "AI screening only; not definitive proof of authenticity."
                }
                st.download_button(
                    "Download audio forensic report (JSON)",
                    json.dumps(report, indent=2),
                    file_name="deeptrace_audio_report.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Audio analysis failed: {e}")
                st.info(
                    "Check HF_TOKEN in Streamlit Secrets, that best_model.onnx exists "
                    "in your Hugging Face repository, and that dependencies are installed."
                )
            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass


st.divider()
st.caption(
    "DEEP-TRACE is an AI screening tool. Predictions can be wrong "
    "and should not be treated as definitive proof."
)
