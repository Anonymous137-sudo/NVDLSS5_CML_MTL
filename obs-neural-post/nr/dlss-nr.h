#pragma once
#include <stdint.h>
#include <stdbool.h>
#define NR_ABI_VERSION 1u
uint32_t nr_abi_version(void);
void nr_set_metallib(const char *path);
void *nr_create(const char *json);
bool nr_update(void *handle,const char *json);
void nr_destroy(void *handle);
void nr_submit(void *handle,void *pixels,uint64_t sequence,double pts);
void *nr_frame_acquire(void *handle);
void *nr_frame_pixels(void *frame);
uint64_t nr_frame_sequence(void *frame); // Unique presentation packet ID, including generated phases.
int32_t nr_frame_kind(void *frame); //0 base,1 synthetic,2 capture-guided reconstruction.
void nr_frame_release(void *frame);
void *nr_handle_retain(void *handle);
void nr_handle_release(void *handle);
void nr_status(void *handle,char *text,int32_t size);
void nr_set_status_overlay(void *handle,bool enabled);
void nr_show_preview(void *handle);
