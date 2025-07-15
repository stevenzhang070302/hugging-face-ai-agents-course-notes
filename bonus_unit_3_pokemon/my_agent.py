STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        ...
    },
    "choose_switch": {
        ...
    },
}

class LLMAgentBase(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_tools = STANDARD_TOOL_SCHEMA
        self.battle_history = []

    def _format_battle_state(self, battle: Battle) -> str:
        active_pkmn = battle.active_pokemon
        active_pkmn_info = f"Your active Pokemon: {active_pkmn.species} " \
                           f"(Type: {'/'.join(map(str, active_pkmn.types))}) " \
                           f"HP: {active_pkmn.current_hp_fraction * 100:.1f}% " \
                           f"Status: {active_pkmn.status.name if active_pkmn.status else 'None'} " \
                           f"Boosts: {active_pkmn.boosts}"

        opponent_pkmn = battle.opponent_active_pokemon
        opp_info_str = "Unknown"
        if opponent_pkmn:
            opp_info_str = f"{opponent_pkmn.species} " \
                           f"(Type: {'/'.join(map(str, opponent_pkmn.types))}) " \
                           f"HP: {opponent_pkmn.current_hp_fraction * 100:.1f}% " \
                           f"Status: {opponent_pkmn.status.name if opponent_pkmn.status else 'None'} " \
                           f"Boosts: {opponent_pkmn.boosts}"
        opponent_pkmn_info = f"Opponent's active Pokemon: {opp_info_str}"

        available_moves_info = "Available moves:\n"
        if battle.available_moves:
            available_moves_info += "\n".join(
                [f"- {move.id} (Type: {move.type}, BP: {move.base_power}, Acc: {move.accuracy}, PP: {move.current_pp}/{move.max_pp}, Cat: {move.category.name})"
                 for move in battle.available_moves]
            )
        else:
             available_moves_info += "- None (Must switch or Struggle)"

        available_switches_info = "Available switches:\n"
        if battle.available_switches:
              available_switches_info += "\n".join(
                  [f"- {pkmn.species} (HP: {pkmn.current_hp_fraction * 100:.1f}%, Status: {pkmn.status.name if pkmn.status else 'None'})"
                   for pkmn in battle.available_switches]
              )
        else:
            available_switches_info += "- None"

        state_str = f"{active_pkmn_info}\n" \
                    f"{opponent_pkmn_info}\n\n" \
                    f"{available_moves_info}\n\n" \
                    f"{available_switches_info}\n\n" \
                    f"Weather: {battle.weather}\n" \
                    f"Terrains: {battle.fields}\n" \
                    f"Your Side Conditions: {battle.side_conditions}\n" \
                    f"Opponent Side Conditions: {battle.opponent_side_conditions}"
        return state_str.strip()

    def _find_move_by_name(self, battle: Battle, move_name: str) -> Optional[Move]:
        normalized_name = normalize_name(move_name)
        # Prioritize exact ID match
        for move in battle.available_moves:
            if move.id == normalized_name:
                return move
        # Fallback: Check display name (less reliable)
        for move in battle.available_moves:
            if move.name.lower() == move_name.lower():
                print(f"Warning: Matched move by display name '{move.name}' instead of ID '{move.id}'. Input was '{move_name}'.")
                return move
        return None

    def _find_pokemon_by_name(self, battle: Battle, pokemon_name: str) -> Optional[Pokemon]:
        normalized_name = normalize_name(pokemon_name)
        for pkmn in battle.available_switches:
            # Normalize the species name for comparison
            if normalize_name(pkmn.species) == normalized_name:
                return pkmn
        return None

    async def choose_move(self, battle: Battle) -> str:
        battle_state_str = self._format_battle_state(battle)
        decision_result = await self._get_llm_decision(battle_state_str)
        print(decision_result)
        decision = decision_result.get("decision")
        error_message = decision_result.get("error")
        action_taken = False
        fallback_reason = ""

        if decision:
            function_name = decision.get("name")
            args = decision.get("arguments", {})
            if function_name == "choose_move":
                move_name = args.get("move_name")
                if move_name:
                    chosen_move = self._find_move_by_name(battle, move_name)
                    if chosen_move and chosen_move in battle.available_moves:
                        action_taken = True
                        chat_msg = f"AI Decision: Using move '{chosen_move.id}'."
                        print(chat_msg)
                        return self.create_order(chosen_move)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid move '{move_name}'."
                else:
                     fallback_reason = "LLM 'choose_move' called without 'move_name'."
            elif function_name == "choose_switch":
                pokemon_name = args.get("pokemon_name")
                if pokemon_name:
                    chosen_switch = self._find_pokemon_by_name(battle, pokemon_name)
                    if chosen_switch and chosen_switch in battle.available_switches:
                        action_taken = True
                        chat_msg = f"AI Decision: Switching to '{chosen_switch.species}'."
                        print(chat_msg)
                        return self.create_order(chosen_switch)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid switch '{pokemon_name}'."
                else:
                    fallback_reason = "LLM 'choose_switch' called without 'pokemon_name'."
            else:
                fallback_reason = f"LLM called unknown function '{function_name}'."

        if not action_taken:
            if not fallback_reason:
                 if error_message:
                     fallback_reason = f"API Error: {error_message}"
                 elif decision is None:
                      fallback_reason = "LLM did not provide a valid function call."
                 else:
                      fallback_reason = "Unknown error processing LLM decision."

            print(f"Warning: {fallback_reason} Choosing random action.")

            if battle.available_moves or battle.available_switches:
                 return self.choose_random_move(battle)
            else:
                 print("AI Fallback: No moves or switches available. Using Struggle/Default.")
                 return self.choose_default_move(battle)

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _get_llm_decision")
    

class TemplateAgent(LLMAgentBase):
    """Uses Template AI API for decisions."""
    def __init__(self, api_key: str = None, model: str = "model-name", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.template_client = TemplateModelProvider(api_key=...)
        self.template_tools = list(self.standard_tools.values())

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        """Sends state to the LLM and gets back the function call decision."""
        system_prompt = (
            "You are a ..."
        )
        user_prompt = f"..."

        try:
            response = await self.template_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            message = response.choices[0].message
            
            return {"decision": {"name": function_name, "arguments": arguments}}

        except Exception as e:
            print(f"Unexpected error during call: {e}")
            return {"error": f"Unexpected error: {e}"}