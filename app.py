import torch

import gradio as gr
from transformers import pipeline

asr = pipeline(
    task="automatic-speech-recognition",
    model="whispy/whisper_hf",
    chunk_length_s=30,
    device="cpu",
)


def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
                "ATTENZIONE: hai caricato un file audio e hai utilizzato il microfono."
                "Il file registrato dal microfono verrà utilizzato e l'audio caricato verrà scartato.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERRORE: Si prega di usare il microfono oppure fare l'upload di un file audio"

    file = microphone if microphone is not None else file_upload

    text = asr(file)["text"]

    return warn_output + text


def main():

    transcribe_interface = gr.Interface(
            fn=transcribe,
            inputs=[
                gr.Audio(sources=["microphone"], type="filepath"),
                gr.Audio(sources=["upload"], type="filepath"),
                ],
            outputs=[
                gr.Textbox(label="Trascrizione del testo"),
                ],
            theme="huggingface",
            title="WhisperIta: Una demo per fare la trascrizione del linguaggio parlato in Italiano",
            description=(
                "Trascrivi file audio o parla direttamente nel microfono semplicemente premendo un pulsante!"
                ),
            allow_flagging="never",
            )

    transcribe_interface.launch()


if __name__ == "__main__":
    main()
