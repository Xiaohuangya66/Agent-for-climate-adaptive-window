"""Same spectral/output assembly used for the TMYx training-label campaign."""
import re
from src.vendor.thermo_agent.energyplus import GOAL2_AUDIT_OUTPUT_REQUESTS,glazing_spectral_block


def idf_objects(text):
    end=0
    for match in re.finditer(r'(?:[^!;]|![^\n]*(?:\n|$))*;',text):
        raw=match.group()
        fields=[x.strip() for x in re.sub(r'![^\n]*','',raw[:-1]).split(',')]
        end=match.end()
        if fields and fields[0]:yield raw,fields
    if re.sub(r'![^\n]*','',text[end:]).strip():raise ValueError('Unterminated IDF object')


def build_idf_text(base,cold,hot):
    if cold is None or hot is None:raise ValueError('Both spectral states are required')
    output=[];seen=set();variable_requests=set()
    for raw,fields in idf_objects(base):
        key=tuple(x.casefold() for x in fields)
        if key[0]=='materialproperty:glazingspectraldata' and key[1] in ['vo2_cold_spec','vo2_hot_spec']:continue
        if key[0] in ['output:meter','output:variable']:
            if key in seen:continue
            seen.add(key)
            if key[0]=='output:variable' and len(key)>=4:variable_requests.add((key[2],key[3]))
        output.append(raw)
    required=list(idf_objects(GOAL2_AUDIT_OUTPUT_REQUESTS))
    required+=list(idf_objects('Output:Variable,*,Daylighting Reference Point 1 Illuminance,Hourly;'))
    for raw,fields in required:
        key=tuple(x.casefold() for x in fields)
        if key in seen:continue
        if key[0]=='output:variable' and (key[2],key[3]) in variable_requests:continue
        output.append('\n'+raw);seen.add(key)
        if key[0]=='output:variable':variable_requests.add((key[2],key[3]))
    output+=['\n'+glazing_spectral_block('VO2_Cold_Spec',cold),'\n'+glazing_spectral_block('VO2_Hot_Spec',hot)]
    return ''.join(output)+'\n'
