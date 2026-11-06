import logging

from livekit.agents import (
    Agent,
    RunContext,
    function_tool,
)

logger = logging.getLogger("support_agent")


class SupportAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Você é um agente de suporte de um aplicativo de transporte. "
                "Ajude os clientes com reclamações, cobranças e segurança. "
                "Se encontrar um alerta [POLICY: ...], siga suas orientações. "
                "Responda sempre em português brasileiro."
            )
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Cumprimente o cliente e pergunte como pode ajudá-lo."
            )
        )

    @function_tool
    async def escalate_to_safety_team(
        self, context: RunContext
    ) -> str:
        """
        Utilize esta ferramenta quando o cliente estiver em perigo
        imediato ou quando o contexto contiver um alerta
        [POLICY: SAFETY EMERGENCY].
        """
        logger.warning(
            f"[ESCALATION] ride={context.userdata.ride_id}, "
            f"rider={context.userdata.rider_name}"
        )

        context.userdata.safety_emergency = True

        return (
            "Esta chamada foi encaminhada para nossa equipe de "
            "segurança, disponível 24 horas por dia. "
            "Um especialista entrará em contato com o cliente "
            "em até dois minutos. "
            "Permaneça na linha até que o cliente confirme "
            "que está em segurança."
        )