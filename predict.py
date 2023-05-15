from typing import Optional

from cog import BaseModel, BasePredictor, Input, Path
from scipy.io.wavfile import write as write_wav

from bark import SAMPLE_RATE, preload_models, save_as_prompt
from bark.api import semantic_to_waveform
from bark.generation import ALLOWED_PROMPTS, generate_text_semantic


class ModelOutput(BaseModel):
    prompt_npz: Optional[Path]
    audio_out: Path


class Predictor(BasePredictor):
    def setup(self):
        # for the pushed version on Replicate, the CACHE_DIR from bark/generation.py is changed to a local folder to
        # include the weights file in the image for faster inference
        preload_models()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Hello, my name is Colin. I like making things, and thinking about things that other people have made.",
        ),
        history_prompt: str = Input(
            description="Voice (from history)",
            default=None,
            choices=sorted(list(ALLOWED_PROMPTS)),
        ),
        custom_history_prompt: Path = Input(
            description="To clone, provide an .npz file. Overrides history_prompt",
            default=None,
        ),
        text_temp: float = Input(
            description="Text temperature (0.0 = conservative, 1.0 = diverse",
            default=0.7,
        ),
        waveform_temp: float = Input(
            description="Waveform temperature (0.0 = conservative, 1.0 = diverse",
            default=0.7,
        ),
        min_eos_p: float = Input(
            description="Lower = earlier termination (helps prevent hallucinations at the end of output)",
            default=0.05,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model. Use min_eos_p to try and control hallucinations."""

        if custom_history_prompt is not None:
            history_prompt = str(custom_history_prompt)

        semantic_tokens = generate_text_semantic(
            prompt, history_prompt=history_prompt, temp=text_temp, min_eos_p=min_eos_p
        )

        audio_array = semantic_to_waveform(
            semantic_tokens=semantic_tokens,
            temp=waveform_temp,
            history_prompt=history_prompt,
            output_full=True,
        )

        output = "/tmp/audio.wav"
        out_npz = "/tmp/prompt.npz"

        save_as_prompt(out_npz, audio_array[0])
        write_wav(output, SAMPLE_RATE, audio_array[-1])
        return ModelOutput(prompt_npz=Path(out_npz), audio_out=Path(output))
