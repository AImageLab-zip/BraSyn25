import json, synapseclient

syn = synapseclient.Synapse()
token = 'your_super_secret_synapse_authentication_token'
syn.login(authToken=token)
print('authentication was successful')
synid = "syn68825416" #this is right
commit = {
   "tag": "just_a_tag",
   "digest": "the_ghcr_digest"
}
syn.restPOST(f"/entity/{synid}/dockerCommit", json.dumps(commit))