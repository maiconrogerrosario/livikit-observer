import logging
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
)

from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from support_agent import SupportAgent
from policy_observer import PolicyObserver




load_dotenv()

logger = logging.getLogger("rideshare-agent")

@dataclass
class RideData:
    ride_id: str | None = None
    rider_name: str | None = None
    pickup_location: str | None = None
    destination: str | None = None
    status: str = "searching"
    safety_emergency: bool = False

def start_observer(session: AgentSession, observer_llm) -> PolicyObserver:
    """Attach a PolicyObserver to the session. Returns immediately, non-blocking."""
    return PolicyObserver(session=session, observer_llm=observer_llm)


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm



@server.rtc_session(agent_name="rideshare-agent")
async def rideshare_agent(ctx: JobContext) -> None:

    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession[RideData](
        userdata=RideData(),

        stt=openai.STT(
            model="gpt-4o-mini-transcribe",
            language="pt"
        ),

        llm=openai.LLM(
            model="gpt-4.1-mini"
        ),

        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="alloy"
        ),

        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel()
        ),

        vad=ctx.proc.userdata["vad"],
    )

    # LLM independente para monitoramento de políticas.
    observer_llm = openai.LLM(
        model="gpt-4.1"
    )

    # Inicializa o observador sem bloquear a sessão.
    observer = start_observer(
        session=session,
        observer_llm=observer_llm
    )

    # Inicia o agente principal.
    await session.start(
        agent=SupportAgent(),
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
