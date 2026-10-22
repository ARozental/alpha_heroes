#include "StdInc.h"
#include "UpLeftAI.h"

#define NAME "UpLeftAI 0.1"

extern "C" DLL_EXPORT int GetGlobalAiVersion()
{
	return AI_INTERFACE_VER;
}

extern "C" DLL_EXPORT void GetAiName(char * name)
{
	strcpy(name, NAME);
}

extern "C" DLL_EXPORT void GetNewBattleAI(std::shared_ptr<CBattleGameInterface> & out)
{
	out = std::make_shared<UpLeftAI>();
}
