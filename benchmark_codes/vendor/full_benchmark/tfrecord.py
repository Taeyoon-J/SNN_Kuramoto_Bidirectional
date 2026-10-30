"""Minimal raw TF Example bytes-feature decoder; no TensorFlow runtime needed."""
import gzip
import struct


def varint(data,pos):
    value=0;shift=0
    while pos<len(data) and shift<70:
        byte=data[pos];pos+=1;value|=(byte&127)<<shift
        if byte<128:return value,pos
        shift+=7
    raise ValueError('invalid protobuf varint')


def fields(data):
    pos=0
    while pos<len(data):
        tag,pos=varint(data,pos);number,wire=tag>>3,tag&7
        if wire==2:
            size,pos=varint(data,pos);end=pos+size
            if end>len(data):raise ValueError('truncated protobuf field')
            value=data[pos:end];pos=end
        elif wire==0:value,pos=varint(data,pos)
        elif wire in (1,5):
            end=pos+(8 if wire==1 else 4)
            if end>len(data):raise ValueError('truncated fixed field')
            value=data[pos:end];pos=end
        else:raise ValueError('unsupported protobuf wire type')
        yield number,wire,value


def bytes_features(record):
    outer=[v for n,w,v in fields(record) if n==1 and w==2]
    if len(outer)!=1:raise ValueError('one Example.features required')
    result={}
    for n,w,entry in fields(outer[0]):
        if (n,w)!=(1,2):continue
        pair={i:v for i,k,v in fields(entry) if k==2}
        if 1 not in pair or 2 not in pair:raise ValueError('invalid Features map')
        name=pair[1].decode('utf-8');feature=list(fields(pair[2]))
        if len(feature)!=1:raise ValueError('Feature oneof required')
        kind,wire,value=feature[0]
        if (kind,wire)==(1,2):
            # Official MOD stores each uint8 as one bytes-list element. Decode
            # that regular wire representation in C-backed slices, rather than
            # running a Python protobuf loop for every pixel of every image.
            count=len(value)//3
            if len(value)==3*count and value[0::3]==b'\x0a'*count and value[1::3]==b'\x01'*count:
                result[name]=[value[2::3]]
            else:result[name]=[v for n,w,v in fields(value) if (n,w)==(1,2)]
    return result


def records(path,limit):
    """Frame boundaries checked; caller binds source to previously verified SHA256."""
    with open(path,'rb') as probe:compressed=probe.read(2)==b'\x1f\x8b'
    opener=gzip.open if compressed else open
    with opener(path,'rb') as stream:
        for index in range(limit):
            header=stream.read(12)
            if len(header)!=12:raise ValueError(f'truncated TFRecord header {index}')
            size=struct.unpack('<Q',header[:8])[0]
            if not 0<size<100_000_000:raise ValueError('unreasonable TFRecord length')
            value=stream.read(size);trailer=stream.read(4)
            if len(value)!=size or len(trailer)!=4:raise ValueError('truncated TFRecord data')
            yield value
