#include "StdInc.h"
#include "AlphaMapAI.h"

#define NAME "AlphaMapAI 0.1"

extern "C" DLL_EXPORT int GetGlobalAiVersion()
{
	return AI_INTERFACE_VER;
}

extern "C" DLL_EXPORT void GetAiName(char * name)
{
	strcpy(name, NAME);
}

extern "C" DLL_EXPORT void GetNewAI(std::shared_ptr<CGlobalAI> & out)
{
	out = std::make_shared<AlphaMapAI>();
}
