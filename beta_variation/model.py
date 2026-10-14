"""
LSTM sequence-to-sequence autoencoder for DMFT data.

Two conditioning modes (selected by use_film):

  use_film=True  — FiLM (Feature-wise Linear Modulation):
      The encoder processes G₀(iω) alone. The label vector is fed through two
      small Dense layers that produce per-unit scale (γ) and shift (β) applied
      directly to the encoder hidden state before it is passed to the decoder.
      This gives the temperature signal a dedicated modulation pathway rather
      than competing with G₀ features inside the LSTM cell.

  use_film=False — concatenation fallback (original behaviour):
      The label is broadcast across the time axis and concatenated with the
      input sequence so the encoder sees the conditioning signal at every step.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, RepeatVector, TimeDistributed,
    concatenate, Dropout, Multiply, Add,
)


def build_lstm_autoencoder(
    sequence_length: int,
    latent_dim: int = 16,
    label_dim: int = 2,
    dropout_rate: float = 0.3,
    use_film: bool = True,
) -> tf.keras.Model:
    """
    Build and compile the LSTM autoencoder.

    Parameters
    ----------
    sequence_length : int
        Number of frequency points (time-steps) per sample.
    latent_dim : int
        Hidden size of both encoder and decoder LSTMs.
    label_dim : int
        Dimension of the conditioning label vector (default 2: [β̃, 1/β̃]).
    dropout_rate : float
        Dropout applied after the encoder hidden state and decoder output.
    use_film : bool
        If True, use FiLM conditioning (recommended).
        If False, fall back to input-concatenation conditioning.

    Inputs
    ------
    ts_input    : (batch, sequence_length, 1)  — Im[G₀(iω)]
    label_input : (batch, label_dim)           — temperature conditioning vector

    Output
    ------
    (batch, sequence_length, 1)  — predicted Im[Σ(iω)]
    """
    ts_input    = Input(shape=(sequence_length, 1), name="ts_input")
    label_input = Input(shape=(label_dim,),          name="label_input")

    if use_film:
        # ── Encoder: G₀ sequence only, no label contamination ────────────────
        _, state_h, state_c = LSTM(
            latent_dim, return_state=True, name="encoder_lstm"
        )(ts_input)

        # ── FiLM: label → γ (scale) and β_shift (shift) ──────────────────────
        # Two independent Dense layers so γ and β_shift can specialise.
        gamma      = Dense(latent_dim, name="film_gamma")(label_input)   # (B, latent_dim)
        beta_shift = Dense(latent_dim, name="film_beta" )(label_input)   # (B, latent_dim)

        # state_h_mod = γ ⊙ state_h + β_shift
        state_h_mod = Add(name="film_shift")(
            [Multiply(name="film_scale")([state_h, gamma]), beta_shift]
        )
        state_h_mod = Dropout(dropout_rate, name="enc_dropout")(state_h_mod)
        encoder_states = [state_h_mod, state_c]

    else:
        # ── Fallback: broadcast label and concatenate with input ──────────────
        label_repeated = RepeatVector(sequence_length)(label_input)       # (B, T, label_dim)
        merged         = concatenate([ts_input, label_repeated], axis=-1) # (B, T, 1+label_dim)

        _, state_h, state_c = LSTM(
            latent_dim, return_state=True, name="encoder_lstm"
        )(merged)
        state_h_mod    = Dropout(dropout_rate, name="enc_dropout")(state_h)
        encoder_states = [state_h_mod, state_c]

    # ── Decoder ───────────────────────────────────────────────────────────────
    decoder_in  = RepeatVector(sequence_length)(state_h_mod)
    decoder_out = LSTM(
        latent_dim, return_sequences=True, name="decoder_lstm"
    )(decoder_in, initial_state=encoder_states)
    decoder_out = Dropout(dropout_rate, name="dec_dropout")(decoder_out)
    output      = TimeDistributed(Dense(1), name="output_projection")(decoder_out)

    model = Model(
        inputs=[ts_input, label_input],
        outputs=output,
        name="lstm_autoencoder",
    )
    model.compile(optimizer="adam", loss="mse")
    return model
